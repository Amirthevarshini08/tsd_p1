import os
import json
import sqlite3
import numpy as np
import re
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import aiohttp
import asyncio
import logging
import base64
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log_handler = logging.getLogger(__name__)

# Constants
DATABASE_FILE = "knowledge_base.db"
MATCH_THRESHOLD = 0.68
RESULT_LIMIT = 10
load_dotenv()
CONTEXT_PORTIONS = 4
SECRET_KEY = os.getenv("SECRET_KEY")

# Models
class SearchQuery(BaseModel):
    query_text: str
    image_data: Optional[str] = None

class SourceLink(BaseModel):
    link: str
    description: str

class SearchResult(BaseModel):
    response: str
    sources: List[SourceLink]

# Initialize FastAPI application
app = FastAPI(title="Knowledge Search API", description="API for searching the knowledge repository")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Verify secret key is set
if not SECRET_KEY:
    log_handler.error("SECRET_KEY environment variable is missing. Application may fail.")

# Database connection
def establish_db_connection():
    connection = None
    try:
        connection = sqlite3.connect(DATABASE_FILE)
        connection.row_factory = sqlite3.Row
        return connection
    except sqlite3.Error as err:
        error_msg = f"Database connection failed: {str(err)}"
        log_handler.error(error_msg)
        log_handler.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

# Initialize database if it doesn't exist
if not os.path.exists(DATABASE_FILE):
    connection = sqlite3.connect(DATABASE_FILE)
    cursor = connection.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS forum_segments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        thread_id INTEGER,
        category_id INTEGER,
        category_name TEXT,
        post_sequence INTEGER,
        creator TEXT,
        timestamp TEXT,
        reactions INTEGER,
        segment_index INTEGER,
        text_content TEXT,
        link TEXT,
        vector BLOB
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS doc_segments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_name TEXT,
        source_link TEXT,
        retrieved_at TEXT,
        segment_index INTEGER,
        text_content TEXT,
        vector BLOB
    )
    ''')
    connection.commit()
    connection.close()

# Calculate vector similarity
def vector_match_score(vector_a, vector_b):
    try:
        vector_a = np.array(vector_a)
        vector_b = np.array(vector_b)
        
        if np.all(vector_a == 0) or np.all(vector_b == 0):
            return 0.0
            
        dot_product = np.dot(vector_a, vector_b)
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return dot_product / (norm_a * norm_b)
    except Exception as err:
        log_handler.error(f"Vector match score error: {err}")
        log_handler.error(traceback.format_exc())
        return 0.0

# Get vector representation with retry logic
async def fetch_vector_representation(content, max_attempts=3):
    if not SECRET_KEY:
        error_msg = "SECRET_KEY environment variable missing"
        log_handler.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    
    attempts = 0
    while attempts < max_attempts:
        try:
            log_handler.info(f"Fetching vector for content (length: {len(content)})")
            endpoint = "https://aipipe.org/openai/v1/embeddings"
            headers = {
                "Authorization": SECRET_KEY,
                "Content-Type": "application/json"
            }
            data = {
                "model": "text-embedding-3-small",
                "input": content
            }
            
            log_handler.info("Requesting vector from API")
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        log_handler.info("Vector received successfully")
                        return result["data"][0]["embedding"]
                    elif response.status == 429:
                        error_text = await response.text()
                        log_handler.warning(f"Rate limit hit, retrying (attempt {attempts+1}): {error_text}")
                        await asyncio.sleep(5 * (attempts + 1))
                        attempts += 1
                    else:
                        error_text = await response.text()
                        error_msg = f"Vector fetch error (status {response.status}): {error_text}"
                        log_handler.error(error_msg)
                        raise HTTPException(status_code=response.status, detail=error_msg)
        except Exception as err:
            error_msg = f"Vector fetch exception (attempt {attempts+1}/{max_attempts}): {err}"
            log_handler.error(error_msg)
            log_handler.error(traceback.format_exc())
            attempts += 1
            if attempts >= max_attempts:
                raise HTTPException(status_code=500, detail=error_msg)
            await asyncio.sleep(3 * attempts)

# Search for matching content
async def locate_matching_content(search_vector, connection):
    try:
        log_handler.info("Searching for matching content")
        cursor = connection.cursor()
        matches = []
        
        log_handler.info("Querying forum segments")
        cursor.execute("""
        SELECT id, thread_id, category_id, category_name, post_sequence, creator, 
               timestamp, reactions, segment_index, text_content, link, vector 
        FROM forum_segments 
        WHERE vector IS NOT NULL
        """)
        
        forum_segments = cursor.fetchall()
        log_handler.info(f"Processing {len(forum_segments)} forum segments")
        processed = 0
        
        for segment in forum_segments:
            try:
                vector = json.loads(segment["vector"])
                score = vector_match_score(search_vector, vector)
                
                if score >= MATCH_THRESHOLD:
                    link = segment["link"]
                    if not link.startswith("http"):
                        link = f"https://discourse.onlinedegree.iitm.ac.in/t/{link}"
                    
                    matches.append({
                        "origin": "forum",
                        "id": segment["id"],
                        "thread_id": segment["thread_id"],
                        "category_id": segment["category_id"],
                        "title": segment["category_name"],
                        "link": link,
                        "content": segment["text_content"],
                        "creator": segment["creator"],
                        "timestamp": segment["timestamp"],
                        "segment_index": segment["segment_index"],
                        "score": float(score)
                    })
                
                processed += 1
                if processed % 1000 == 0:
                    log_handler.info(f"Processed {processed}/{len(forum_segments)} forum segments")
                    
            except Exception as err:
                log_handler.error(f"Error processing forum segment {segment['id']}: {err}")
        
        log_handler.info("Querying document segments")
        cursor.execute("""
        SELECT id, document_name, source_link, retrieved_at, segment_index, text_content, vector 
        FROM doc_segments 
        WHERE vector IS NOT NULL
        """)
        
        doc_segments = cursor.fetchall()
        log_handler.info(f"Processing {len(doc_segments)} document segments")
        processed = 0
        
        for segment in doc_segments:
            try:
                vector = json.loads(segment["vector"])
                score = vector_match_score(search_vector, vector)
                
                if score >= MATCH_THRESHOLD:
                    link = segment["source_link"]
                    if not link or not link.startswith("http"):
                        link = f"https://docs.onlinedegree.iitm.ac.in/{segment['document_name']}"
                    
                    matches.append({
                        "origin": "document",
                        "id": segment["id"],
                        "title": segment["document_name"],
                        "link": link,
                        "content": segment["text_content"],
                        "segment_index": segment["segment_index"],
                        "score": float(score)
                    })
                
                processed += 1
                if processed % 1000 == 0:
                    log_handler.info(f"Processed {processed}/{len(doc_segments)} document segments")
                    
            except Exception as err:
                log_handler.error(f"Error processing document segment {segment['id']}: {err}")
        
        matches.sort(key=lambda x: x["score"], reverse=True)
        log_handler.info(f"Found {len(matches)} relevant matches")
        
        grouped_matches = {}
        
        for match in matches:
            key = f"{match['origin']}_{match['thread_id' if match['origin'] == 'forum' else 'title']}"
            if key not in grouped_matches:
                grouped_matches[key] = []
            
            grouped_matches[key].append(match)
        
        final_matches = []
        for key, segments in grouped_matches.items():
            segments.sort(key=lambda x: x["score"], reverse=True)
            final_matches.extend(segments[:CONTEXT_PORTIONS])
        
        final_matches.sort(key=lambda x: x["score"], reverse=True)
        log_handler.info(f"Returning {len(final_matches[:RESULT_LIMIT])} final matches")
        return final_matches[:RESULT_LIMIT]
    except Exception as err:
        error_msg = f"Error in locate_matching_content: {err}"
        log_handler.error(error_msg)
        log_handler.error(traceback.format_exc())
        raise

# Enrich matches with surrounding content
async def enhance_with_context(connection, matches):
    try:
        log_handler.info(f"Enhancing {len(matches)} matches with context")
        cursor = connection.cursor()
        enhanced_matches = []
        
        for match in matches:
            enhanced_match = match.copy()
            extra_content = ""
            
            if match["origin"] == "forum":
                thread_id = match["thread_id"]
                current_index = match["segment_index"]
                
                if current_index > 0:
                    cursor.execute("""
                    SELECT text_content FROM forum_segments 
                    WHERE thread_id = ? AND segment_index = ?
                    """, (thread_id, current_index - 1))
                    prev_segment = cursor.fetchone()
                    if prev_segment:
                        extra_content = prev_segment["text_content"] + " "
                
                cursor.execute("""
                SELECT text_content FROM forum_segments 
                WHERE thread_id = ? AND segment_index = ?
                """, (thread_id, current_index + 1))
                next_segment = cursor.fetchone()
                if next_segment:
                    extra_content += " " + next_segment["text_content"]
                
            elif match["origin"] == "document":
                title = match["title"]
                current_index = match["segment_index"]
                
                if current_index > 0:
                    cursor.execute("""
                    SELECT text_content FROM doc_segments 
                    WHERE document_name = ? AND segment_index = ?
                    """, (title, current_index - 1))
                    prev_segment = cursor.fetchone()
                    if prev_segment:
                        extra_content = prev_segment["text_content"] + " "
                
                cursor.execute("""
                SELECT text_content FROM doc_segments 
                WHERE document_name = ? AND segment_index = ?
                """, (title, current_index + 1))
                next_segment = cursor.fetchone()
                if next_segment:
                    extra_content += " " + next_segment["text_content"]
            
            if extra_content:
                enhanced_match["content"] = f"{match['content']} {extra_content}"
            
            enhanced_matches.append(enhanced_match)
        
        log_handler.info(f"Enhanced {len(enhanced_matches)} matches")
        return enhanced_matches
    except Exception as err:
        error_msg = f"Error in enhance_with_context: {err}"
        log_handler.error(error_msg)
        log_handler.error(traceback.format_exc())
        raise

# Generate response using language model
async def create_response(query, relevant_matches, max_attempts=2):
    if not SECRET_KEY:
        error_msg = "SECRET_KEY environment variable missing"
        log_handler.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    
    attempts = 0
    while attempts < max_attempts:    
        try:
            log_handler.info(f"Creating response for query: '{query[:50]}...'")
            context = ""
            for match in relevant_matches:
                source_type = "Forum post" if match["origin"] == "forum" else "Documentation"
                context += f"\n\n{source_type} (Link: {match['link']}):\n{match['content'][:1500]}"
            
            prompt = f"""Respond to the query using ONLY the provided context. 
            If the context is insufficient, state "I lack sufficient information to respond to this query."
            
            Context:
            {context}
            
            Query: {query}
            
            Format your response as:
            1. A complete but concise response
            2. A "References:" section listing the links and relevant snippets used
            
            References format:
            References:
            1. Link: [exact_link_1], Description: [brief excerpt]
            2. Link: [exact_link_2], Description: [brief excerpt]
            
            Use exact links from the context without modifications.
            """
            
            log_handler.info("Requesting response from LLM API")
            endpoint = "https://aipipe.org/openai/v1/chat/completions"
            headers = {
                "Authorization": SECRET_KEY,
                "Content-Type": "application/json"
            }
            data = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant providing accurate responses based solely on given context. Always include exact links in references."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        log_handler.info("Response received from LLM")
                        return result["choices"][0]["message"]["content"]
                    elif response.status == 429:
                        error_text = await response.text()
                        log_handler.warning(f"Rate limit hit, retrying (attempt {attempts+1}): {error_text}")
                        await asyncio.sleep(3 * (attempts + 1))
                        attempts += 1
                    else:
                        error_text = await response.text()
                        error_msg = f"Response generation error (status {response.status}): {error_text}"
                        log_handler.error(error_msg)
                        raise HTTPException(status_code=response.status, detail=error_msg)
        except Exception as err:
            error_msg = f"Response generation exception: {err}"
            log_handler.error(error_msg)
            log_handler.error(traceback.format_exc())
            attempts += 1
            if attempts >= max_attempts:
                raise HTTPException(status_code=500, detail=error_msg)
            await asyncio.sleep(2)

# Process query with optional image
async def handle_multimodal_input(query, image_data):
    if not SECRET_KEY:
        error_msg = "SECRET_KEY environment variable missing"
        log_handler.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
        
    try:
        log_handler.info(f"Processing query: '{query[:50]}...', image: {image_data is not None}")
        if not image_data:
            log_handler.info("No image, processing text-only")
            return await fetch_vector_representation(query)
        
        log_handler.info("Processing query with image")
        endpoint = "https://aipipe.org/openai/v1/chat/completions"
        headers = {
            "Authorization": SECRET_KEY,
            "Content-Type": "application/json"
        }
        
        image_content = f"data:image/jpeg;base64,{image_data}"
        
        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Analyze this image in relation to: {query}"},
                        {"type": "image_url", "image_url": {"url": image_content}}
                    ]
                }
            ]
        }
        
        log_handler.info("Requesting vision API response")
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    image_description = result["choices"][0]["message"]["content"]
                    log_handler.info(f"Image description: '{image_description[:50]}...'")
                    
                    combined_input = f"{query}\nImage details: {image_description}"
                    return await fetch_vector_representation(combined_input)
                else:
                    error_text = await response.text()
                    log_handler.error(f"Image processing error (status {response.status}): {error_text}")
                    log_handler.info("Falling back to text-only processing")
                    return await fetch_vector_representation(query)
    except Exception as err:
        log_handler.error(f"Multimodal processing exception: {err}")
        log_handler.error(traceback.format_exc())
        log_handler.info("Falling back to text-only processing")
        return await fetch_vector_representation(query)

# Parse language model output
def extract_response_content(output):
    try:
        log_handler.info("Extracting response content")
        
        parts = output.split("References:", 1)
        if len(parts) == 1:
            for heading in ["Reference:", "Sources:", "Source:"]:
                if heading in output:
                    parts = output.split(heading, 1)
                    break
        
        response_text = parts[0].strip()
        sources = []
        
        if len(parts) > 1:
            references_text = parts[1].strip()
            reference_lines = references_text.split("\n")
            
            for line in reference_lines:
                line = line.strip()
                if not line:
                    continue
                    
                line = re.sub(r'^\d+\.\s*', '', line)
                line = re.sub(r'^-\s*', '', line)
                
                link_match = re.search(r'Link:\s*\[(.*?)\]|link:\s*\[(.*?)\]|\[(http[^\]]+)\]|Link:\s*(http\S+)|link:\s*(http\S+)|(http\S+)', line, re.IGNORECASE)
                desc_match = re.search(r'Description:\s*\[(.*?)\]|description:\s*\[(.*?)\]|[""](.*?)[""]|Description:\s*"(.*?)"|description:\s*"(.*?)"', line, re.IGNORECASE)
                
                if link_match:
                    link = next((g for g in link_match.groups() if g), "")
                    link = link.strip()
                    
                    description = "Source reference"
                    
                    if desc_match:
                        desc_value = next((g for g in desc_match.groups() if g), "")
                        if desc_value:
                            description = desc_value.strip()
                    
                    if link and link.startswith("http"):
                        sources.append({"link": link, "description": description})
        
        log_handler.info(f"Extracted response (length: {len(response_text)}), {len(sources)} sources")
        return {"response": response_text, "sources": sources}
    except Exception as err:
        error_msg = f"Response parsing error: {err}"
        log_handler.error(error_msg)
        log_handler.error(traceback.format_exc())
        return {
            "response": "Failed to parse language model response.",
            "sources": []
        }

# API endpoints
@app.post("/search")
async def search_knowledge_repository(request: SearchQuery):
    try:
        log_handler.info(f"Received search: query='{request.question[:50]}...', image={request.image is not None}")
        
        if not SECRET_KEY:
            error_msg = "SECRET_KEY environment variable missing"
            log_handler.error(error_msg)
            return JSONResponse(
                status_code=500,
                content={"error": error_msg}
            )
            
        connection = establish_db_connection()
        
        try:
            log_handler.info("Generating query vector")
            search_vector = await handle_multimodal_input(
                request.question,
                request.image
            )
            
            log_handler.info("Locating matching content")
            relevant_matches = await locate_matching_content(search_vector, connection)
            
            if not relevant_matches:
                log_handler.info("No matches found")
                return {
                    "response": "No relevant information found in the knowledge repository.",
                    "sources": []
                }
            
            log_handler.info("Enhancing matches with context")
            enhanced_matches = await enhance_with_context(connection, relevant_matches)
            
            log_handler.info("Creating response")
            model_output = await create_response(request.query_text, enhanced_matches)
            
            log_handler.info("Extracting response content")
            result = extract_response_content(model_output)
            
            if not result["sources"]:
                log_handler.info("No sources extracted, generating from matches")
                sources = []
                unique_links = set()
                
                for match in relevant_matches[:5]:
                    link = match["link"]
                    if link not in unique_links:
                        unique_links.add(link)
                        snippet = match["content"][:100] + "..." if len(match["content"]) > 100 else match["content"]
                        sources.append({"link": link, "description": snippet})
                
                result["sources"] = sources
            
            log_handler.info(f"Returning result: response_length={len(result['response'])}, sources={len(result['sources'])}")
            return result
        except Exception as err:
            error_msg = f"Search processing error: {err}"
            log_handler.error(error_msg)
            log_handler.error(traceback.format_exc())
            return JSONResponse(
                status_code=500,
                content={"error": error_msg}
            )
        finally:
            connection.close()
    except Exception as err:
        error_msg = f"Unexpected error in search_knowledge_repository: {err}"
        log_handler.error(error_msg)
        log_handler.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": error_msg}
        )

# System status endpoint
@app.get("/status")
async def system_status():
    try:
        connection = sqlite3.connect(DATABASE_FILE)
        cursor = connection.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM forum_segments")
        forum_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM doc_segments")
        doc_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM forum_segments WHERE vector IS NOT NULL")
        forum_vectors = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM doc_segments WHERE vector IS NOT NULL")
        doc_vectors = cursor.fetchone()[0]
        
        connection.close()
        
        return {
            "status": "operational",
            "database": "connected",
            "key_present": bool(SECRET_KEY),
            "forum_segments": forum_count,
            "doc_segments": doc_count,
            "forum_vectors": forum_vectors,
            "doc_vectors": doc_vectors
        }
    except Exception as err:
        log_handler.error(f"System status check failed: {err}")
        return JSONResponse(
            status_code=500,
            content={"status": "failed", "error": str(err), "key_present": bool(SECRET_KEY)}
        )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
