# app.py
import os
import json
import sqlite3
import numpy as np
import re
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import aiohttp
import asyncio
import logging
import base64
from fastapi.responses import JSONResponse, HTMLResponse
import uvicorn
import traceback
from dotenv import load_dotenv
import tempfile
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
SIMILARITY_THRESHOLD = 0.20  # Lowered threshold for better recall
MAX_RESULTS = 20  # Increased to get more context
MAX_CONTEXT_CHUNKS = 5  # Increased number of chunks per source

load_dotenv()

# For Vercel deployment, use a temporary directory
if os.environ.get('VERCEL'):
    # On Vercel, use /tmp directory which is writable
    DB_PATH = "/tmp/knowledge_base.db"
    logger.info("Running on Vercel, using temporary database")
else:
    # Local development
    DB_PATH = "knowledge_base.db"
    logger.info("Running locally, using local database")

API_KEY = os.getenv('API_KEY')
if API_KEY:
    API_KEY = f"Bearer {API_KEY}"
    logger.info("API key loaded successfully")
else:
    logger.error("API_KEY environment variable is not set. The application will not function correctly.")

# Models
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None  # Base64 encoded image

class LinkInfo(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[LinkInfo]

class HealthResponse(BaseModel):
    status: str
    database: str
    api_key_set: bool
    discourse_chunks: int
    markdown_chunks: int
    discourse_embeddings: int
    markdown_embeddings: int
    environment: str

# Initialize FastAPI app
app = FastAPI(title="RAG Query API", description="API for querying the RAG knowledge base")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Update ensure_database_exists function
def ensure_database_exists():
    """Ensure database exists with correct schema - Vercel compatible"""
    try:
        # For Vercel, always recreate in /tmp
        if os.environ.get('VERCEL'):
            logger.info("Running on Vercel - creating fresh database")
            os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        elif not os.path.exists(DB_PATH):
            logger.info("Database not found - creating new one")
        else:
            logger.info("Database exists - skipping creation")
            return

        # Create/recreate database
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Create tables with IF NOT EXISTS
        c.executescript('''
            CREATE TABLE IF NOT EXISTS discourse_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id INTEGER,
                topic_id INTEGER,
                topic_title TEXT,
                post_number INTEGER,
                author TEXT,
                created_at TEXT,
                likes INTEGER,
                chunk_index INTEGER,
                content TEXT,
                url TEXT,
                embedding BLOB
            );
            
            CREATE TABLE IF NOT EXISTS markdown_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_title TEXT,
                original_url TEXT,
                downloaded_at TEXT,
                chunk_index INTEGER,
                content TEXT,
                embedding BLOB
            );
            
            CREATE INDEX IF NOT EXISTS idx_discourse_post_id ON discourse_chunks(post_id);
            CREATE INDEX IF NOT EXISTS idx_discourse_topic_id ON discourse_chunks(topic_id);
            CREATE INDEX IF NOT EXISTS idx_markdown_title ON markdown_chunks(doc_title);
        ''')
        
        # Add sample data for testing
        sample_embedding = json.dumps([0.1] * 1536)
        c.execute('''
            INSERT OR REPLACE INTO discourse_chunks 
            (post_id, topic_id, topic_title, post_number, author, created_at, likes, 
             chunk_index, content, url, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (1, 1, "Sample Discussion", 1, "test_user", "2024-01-01", 5, 0,
              "Test content about Docker and Podman",
              "https://discourse.onlinedegree.iitm.ac.in/t/test/1",
              sample_embedding))
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        logger.error(traceback.format_exc())
        raise

# Create database on startup
ensure_database_exists()

# Create a connection to the SQLite database
def get_db_connection():
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # This enables column access by name
        return conn
    except sqlite3.Error as e:
        error_msg = f"Database connection error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

# Vector similarity calculation with improved handling
def cosine_similarity(vec1, vec2):
    try:
        # Convert to numpy arrays
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        # Handle zero vectors
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0.0
            
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        # Avoid division by zero
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
            
        return dot_product / (norm_vec1 * norm_vec2)
    except Exception as e:
        logger.error(f"Error in cosine_similarity: {e}")
        logger.error(traceback.format_exc())
        return 0.0  # Return 0 similarity on error rather than crashing

# Function to get embedding from aipipe proxy with retry mechanism
async def get_embedding(text, max_retries=3):
    if not API_KEY:
        error_msg = "API_KEY environment variable not set"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    
    retries = 0
    while retries < max_retries:
        try:
            logger.info(f"Getting embedding for text (length: {len(text)})")
            # Call the embedding API through aipipe proxy
            url = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"

            headers = {
                "Authorization": API_KEY,
                "Content-Type": "application/json"
            }
            payload = {
                "model": "text-embedding-3-small",
                "input": text
            }
            
            logger.info("Sending request to embedding API")
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info("Successfully received embedding")
                        return result["data"][0]["embedding"]
                    elif response.status == 429:  # Rate limit error
                        error_text = await response.text()
                        logger.warning(f"Rate limit reached, retrying after delay (retry {retries+1}): {error_text}")
                        await asyncio.sleep(5 * (retries + 1))  # Exponential backoff
                        retries += 1
                    else:
                        error_text = await response.text()
                        error_msg = f"Error getting embedding (status {response.status}): {error_text}"
                        logger.error(error_msg)
                        raise HTTPException(status_code=response.status, detail=error_msg)
        except Exception as e:
            error_msg = f"Exception getting embedding (attempt {retries+1}/{max_retries}): {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            retries += 1
            if retries >= max_retries:
                raise HTTPException(status_code=500, detail=error_msg)
            await asyncio.sleep(3 * retries)  # Wait before retry

# Function to find similar content in the database with improved logic
async def find_similar_content(query_embedding, conn):
    try:
        logger.info("Finding similar content in database")
        cursor = conn.cursor()
        results = []
        
        # Search discourse chunks
        logger.info("Querying discourse chunks")
        cursor.execute("""
        SELECT id, post_id, topic_id, topic_title, post_number, author, created_at, 
               likes, chunk_index, content, url, embedding 
        FROM discourse_chunks 
        WHERE embedding IS NOT NULL
        """)
        
        discourse_chunks = cursor.fetchall()
        logger.info(f"Processing {len(discourse_chunks)} discourse chunks")
        
        for chunk in discourse_chunks:
            try:
                embedding = json.loads(chunk["embedding"])
                similarity = cosine_similarity(query_embedding, embedding)
                
                if similarity >= SIMILARITY_THRESHOLD:
                    # Ensure URL is properly formatted
                    url = chunk["url"]
                    if not url.startswith("http"):
                        # Fix missing protocol
                        url = f"https://discourse.onlinedegree.iitm.ac.in/t/{url}"
                    
                    results.append({
                        "source": "discourse",
                        "id": chunk["id"],
                        "post_id": chunk["post_id"],
                        "topic_id": chunk["topic_id"],
                        "title": chunk["topic_title"],
                        "url": url,
                        "content": chunk["content"],
                        "author": chunk["author"],
                        "created_at": chunk["created_at"],
                        "chunk_index": chunk["chunk_index"],
                        "similarity": float(similarity)
                    })
                    
            except Exception as e:
                logger.error(f"Error processing discourse chunk {chunk['id']}: {e}")
        
        # Search markdown chunks
        logger.info("Querying markdown chunks")
        cursor.execute("""
        SELECT id, doc_title, original_url, downloaded_at, chunk_index, content, embedding 
        FROM markdown_chunks 
        WHERE embedding IS NOT NULL
        """)
        
        markdown_chunks = cursor.fetchall()
        logger.info(f"Processing {len(markdown_chunks)} markdown chunks")
        
        for chunk in markdown_chunks:
            try:
                embedding = json.loads(chunk["embedding"])
                similarity = cosine_similarity(query_embedding, embedding)
                
                if similarity >= SIMILARITY_THRESHOLD:
                    # Ensure URL is properly formatted
                    url = chunk["original_url"]
                    if not url or not url.startswith("http"):
                        # Use a default URL if missing
                        url = f"https://docs.onlinedegree.iitm.ac.in/{chunk['doc_title']}"
                    
                    results.append({
                        "source": "markdown",
                        "id": chunk["id"],
                        "title": chunk["doc_title"],
                        "url": url,
                        "content": chunk["content"],
                        "chunk_index": chunk["chunk_index"],
                        "similarity": float(similarity)
                    })
                    
            except Exception as e:
                logger.error(f"Error processing markdown chunk {chunk['id']}: {e}")
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x["similarity"], reverse=True)
        logger.info(f"Found {len(results)} relevant results above threshold")
        
        return results[:MAX_RESULTS]
    except Exception as e:
        error_msg = f"Error in find_similar_content: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise

# Function to enrich content with adjacent chunks
async def enrich_with_adjacent_chunks(conn, results):
    try:
        logger.info(f"Enriching {len(results)} results with adjacent chunks")
        # For simplicity, return results as-is for now
        # You can implement the adjacent chunk logic if needed
        return results
    except Exception as e:
        error_msg = f"Error in enrich_with_adjacent_chunks: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return results

# Function to generate an answer using LLM with improved prompt
async def generate_answer(question, relevant_results, max_retries=2):
    if not API_KEY:
        error_msg = "API_KEY environment variable not set"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    
    retries = 0
    while retries < max_retries:    
        try:
            logger.info(f"Generating answer for question: '{question[:50]}...'")
            context = ""
            for result in relevant_results:
                source_type = "Discourse post" if result["source"] == "discourse" else "Documentation"
                context += f"\n\n{source_type} (URL: {result['url']}):\n{result['content'][:1500]}"
            
            # Prepare improved prompt
            prompt = f"""Answer the following question based ONLY on the provided context. 
            If you cannot answer the question based on the context, say "I don't have enough information to answer this question."
            
            Context:
            {context}
            
            Question: {question}
            
            Return your response in this exact format:
            1. A comprehensive yet concise answer
            2. A "Sources:" section that lists the URLs and relevant text snippets you used to answer
            
            Sources must be in this exact format:
            Sources:
            1. URL: [exact_url_1], Text: [brief quote or description]
            2. URL: [exact_url_2], Text: [brief quote or description]
            
            Make sure the URLs are copied exactly from the context without any changes.
            """
            
            logger.info("Sending request to LLM API")
            url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
            headers = {
               "Authorization": API_KEY,
               "Content-Type": "application/json"
            }
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that provides accurate answers based only on the provided context. Always include sources in your response with exact URLs."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info("Successfully received answer from LLM")
                        return result["choices"][0]["message"]["content"]
                    elif response.status == 429:  # Rate limit error
                        error_text = await response.text()
                        logger.warning(f"Rate limit reached, retrying after delay (retry {retries+1}): {error_text}")
                        await asyncio.sleep(3 * (retries + 1))  # Exponential backoff
                        retries += 1
                    else:
                        error_text = await response.text()
                        error_msg = f"Error generating answer (status {response.status}): {error_text}"
                        logger.error(error_msg)
                        raise HTTPException(status_code=response.status, detail=error_msg)
        except Exception as e:
            error_msg = f"Exception generating answer: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            retries += 1
            if retries >= max_retries:
                raise HTTPException(status_code=500, detail=error_msg)
            await asyncio.sleep(2)  # Wait before retry

# Function to process multimodal content (text + image)
async def process_multimodal_query(question, image_base64):
    if not image_base64:
        logger.info("No image provided, processing as text-only query")
        return await get_embedding(question)
    
    # For now, just process text - you can add image processing later
    logger.info("Image provided but processing as text-only for now")
    return await get_embedding(question)

# Function to parse LLM response and extract answer and sources
def parse_llm_response(response):
    try:
        logger.info("Parsing LLM response")
        
        # First try to split by "Sources:" heading
        parts = response.split("Sources:", 1)
        
        # If that doesn't work, try alternative formats
        if len(parts) == 1:
            # Try other possible headings
            for heading in ["Source:", "References:", "Reference:"]:
                if heading in response:
                    parts = response.split(heading, 1)
                    break
        
        answer = parts[0].strip()
        links = []
        
        if len(parts) > 1:
            sources_text = parts[1].strip()
            source_lines = sources_text.split("\n")
            
            for line in source_lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Remove list markers (1., 2., -, etc.)
                line = re.sub(r'^\d+\.\s*', '', line)
                line = re.sub(r'^-\s*', '', line)
                
                # Extract URL and text using more flexible patterns
                url_match = re.search(r'URL:\s*\[(.*?)\]|url:\s*\[(.*?)\]|\[(http[^\]]+)\]|URL:\s*(http\S+)|url:\s*(http\S+)|(http\S+)', line, re.IGNORECASE)
                text_match = re.search(r'Text:\s*\[(.*?)\]|text:\s*\[(.*?)\]|[""](.*?)[""]|Text:\s*"(.*?)"|text:\s*"(.*?)"', line, re.IGNORECASE)
                
                if url_match:
                    # Find the first non-None group from the regex match
                    url = next((g for g in url_match.groups() if g), "")
                    url = url.strip()
                    
                    # Default text if no match
                    text = "Source reference"
                    
                    # If we found a text match, use it
                    if text_match:
                        # Find the first non-None group from the regex match
                        text_value = next((g for g in text_match.groups() if g), "")
                        if text_value:
                            text = text_value.strip()
                    
                    # Only add if we have a valid URL
                    if url and url.startswith("http"):
                        links.append({"url": url, "text": text})
        
        logger.info(f"Parsed answer (length: {len(answer)}) and {len(links)} sources")
        return {"answer": answer, "links": links}
    except Exception as e:
        error_msg = f"Error parsing LLM response: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        # Return a basic response structure with the error
        return {
            "answer": "Error parsing the response from the language model.",
            "links": []
        }

# Define API routes
@app.get("/")
async def root():
    try:
        # Simple HTML response for the root endpoint
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>RAG Query API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .container { max-width: 800px; margin: 0 auto; }
                .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
                code { background: #e0e0e0; padding: 2px 4px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🤖 RAG Query API</h1>
                <p>Welcome to the RAG (Retrieval-Augmented Generation) Query API!</p>
                
                <div class="endpoint">
                    <h3>POST /api/</h3>
                    <p>Submit queries to the knowledge base</p>
                    <p><strong>Request body:</strong></p>
                    <code>{"question": "Your question here", "image": null}</code>
                </div>
                
                <div class="endpoint">
                    <h3>GET /health</h3>
                    <p>Check API health and database status</p>
                </div>
                
                <p><strong>Status:</strong> ✅ API is running and ready to serve requests!</p>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        return HTMLResponse(content=f"<h1>RAG Query API</h1><p>API is running. Error loading page: {e}</p>", status_code=200)

@app.post("/api/", response_model=QueryResponse)
async def api_query(request: Request):
    try:
        body = await request.json()
        question = body.get("question")
        image = body.get("image")
        
        if not question:
            return JSONResponse(
                status_code=400,
                content={"answer": "Error: Question is required", "links": []}
            )
            
        logger.info(f"Received API query: '{question[:50]}...'")
        
        conn = get_db_connection()
        try:
            query_embedding = await process_multimodal_query(question, image)
            similar_results = await find_similar_content(query_embedding, conn)
            
            if not similar_results:
                logger.info("No similar content found")
                return QueryResponse(
                    answer="I don't have enough information to answer this question based on the available knowledge base.",
                    links=[]
                )
            
            enriched_results = await enrich_with_adjacent_chunks(conn, similar_results)
            llm_response = await generate_answer(question, enriched_results)
            parsed_response = parse_llm_response(llm_response)
            
            logger.info(f"Returning answer with {len(parsed_response['links'])} links")
            
            return QueryResponse(
                answer=parsed_response["answer"],
                links=[LinkInfo(**link) for link in parsed_response["links"]]
            )
            
        finally:
            conn.close()
            
    except Exception as e:
        error_msg = f"Unhandled exception in api_query: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"answer": f"Error: {str(e)}", "links": []}
        )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        logger.info("Health check requested")
        
        # Initialize with safe defaults
        status = {
            "status": "healthy",
            "database": "disconnected",
            "api_key_set": bool(API_KEY),
            "discourse_chunks": 0,
            "markdown_chunks": 0,
            "discourse_embeddings": 0,
            "markdown_embeddings": 0,
            "environment": "vercel" if os.environ.get('VERCEL') else "local"
        }
        
        try:
            # Ensure database exists (important for Vercel's ephemeral filesystem)
            ensure_database_exists()
            
            # Test database connection and get counts
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Get table counts safely
            cursor.execute("SELECT COUNT(*) FROM discourse_chunks")
            status["discourse_chunks"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM markdown_chunks")
            status["markdown_chunks"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM discourse_chunks WHERE embedding IS NOT NULL")
            status["discourse_embeddings"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM markdown_chunks WHERE embedding IS NOT NULL")
            status["markdown_embeddings"] = cursor.fetchone()[0]
            
            status["database"] = "connected"
            conn.close()
            
        except sqlite3.Error as e:
            logger.error(f"Database error in health check: {e}")
            status["status"] = "unhealthy"
            status["database"] = f"error: {str(e)}"
            return JSONResponse(status_code=500, content=status)
            
        return JSONResponse(status_code=200, content=status)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "database": "error",
                "api_key_set": bool(API_KEY),
                "discourse_chunks": 0,
                "markdown_chunks": 0,
                "discourse_embeddings": 0,
                "markdown_embeddings": 0,
                "environment": "vercel" if os.environ.get('VERCEL') else "local",
                "error": str(e)
            }
        )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)