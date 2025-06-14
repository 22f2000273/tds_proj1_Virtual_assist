# app.py
import os
import json
import sqlite3
import numpy as np
import re
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import aiohttp
import asyncio
import logging
import traceback
from fastapi.responses import JSONResponse, HTMLResponse
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
SIMILARITY_THRESHOLD = 0.20
MAX_RESULTS = 20
MAX_CONTEXT_CHUNKS = 5

# Get API key with better error handling
API_KEY = os.getenv('API_KEY')
if not API_KEY:
    logger.error("API_KEY environment variable is not set. The application will not function correctly.")
    # Don't raise an error immediately, let the app start but show proper error messages
    API_KEY = None

# Ensure Bearer prefix is added only once if API_KEY exists
if API_KEY and not API_KEY.startswith('Bearer '):
    API_KEY = f"Bearer {API_KEY}"

# Models
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None

class LinkInfo(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[LinkInfo]

# Initialize FastAPI app
app = FastAPI(title="RAG Query API", description="API for querying the RAG knowledge base")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Database connection with proper error handling for Vercel
def get_db_connection():
    """
    For Vercel deployment, you'll need to replace this with a proper database service
    like Vercel Postgres, PlanetScale, or Supabase
    """
    try:
        # This will fail on Vercel - you need to use a cloud database
        db_path = os.getenv('DB_PATH', '/tmp/knowledge_base.db')  # Use /tmp for temporary files
        
        # Check if we're in Vercel environment
        if os.getenv('VERCEL'):
            # You should replace this with actual cloud database connection
            raise HTTPException(
                status_code=500, 
                detail="Database not configured for production. Please set up Vercel Postgres or another cloud database."
            )
        
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        error_msg = f"Database connection error: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

# Initialize database (this won't work on Vercel)
def init_database():
    """Initialize database - only works locally"""
    if os.getenv('VERCEL'):
        return  # Skip database initialization on Vercel
        
    db_path = os.getenv('DB_PATH', 'knowledge_base.db')
    if not os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        # Create discourse_chunks table
        c.execute('''
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
        )
        ''')
        
        # Create markdown_chunks table
        c.execute('''
        CREATE TABLE IF NOT EXISTS markdown_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_title TEXT,
            original_url TEXT,
            downloaded_at TEXT,
            chunk_index INTEGER,
            content TEXT,
            embedding BLOB
        )
        ''')
        conn.commit()
        conn.close()

# Initialize database
init_database()

# Vector similarity calculation
def cosine_similarity(vec1, vec2):
    try:
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0.0
            
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
            
        return dot_product / (norm_vec1 * norm_vec2)
    except Exception as e:
        logger.error(f"Error in cosine_similarity: {e}")
        return 0.0

# Function to get embedding from API
async def get_embedding(text, max_retries=3):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API_KEY not configured. Please set the API_KEY environment variable.")
    
    retries = 0
    while retries < max_retries:
        try:
            logger.info(f"Getting embedding for text (length: {len(text)})")
            
            url = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"
            headers = {
                "Authorization": API_KEY,
                "Content-Type": "application/json"
            }
            payload = {
                "model": "text-embedding-3-small",
                "input": text
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=30) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info("Successfully received embedding")
                        return result["data"][0]["embedding"]
                    elif response.status == 429:
                        error_text = await response.text()
                        logger.warning(f"Rate limit reached, retrying after delay (retry {retries+1}): {error_text}")
                        await asyncio.sleep(5 * (retries + 1))
                        retries += 1
                    else:
                        error_text = await response.text()
                        error_msg = f"Error getting embedding (status {response.status}): {error_text}"
                        logger.error(error_msg)
                        raise HTTPException(status_code=response.status, detail=error_msg)
        except asyncio.TimeoutError:
            logger.warning(f"Timeout getting embedding (retry {retries+1})")
            retries += 1
            if retries >= max_retries:
                raise HTTPException(status_code=500, detail="Timeout getting embedding")
            await asyncio.sleep(3 * retries)
        except Exception as e:
            error_msg = f"Exception getting embedding (attempt {retries+1}/{max_retries}): {e}"
            logger.error(error_msg)
            retries += 1
            if retries >= max_retries:
                raise HTTPException(status_code=500, detail=error_msg)
            await asyncio.sleep(3 * retries)

# Function to find similar content
async def find_similar_content(query_embedding, conn):
    try:
        logger.info("Finding similar content in database")
        cursor = conn.cursor()
        results = []
        
        # Search discourse chunks
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
                    url = chunk["url"]
                    if not url.startswith("http"):
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
                    url = chunk["original_url"]
                    if not url or not url.startswith("http"):
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
        
        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        logger.info(f"Found {len(results)} relevant results above threshold")
        
        # Group by source document
        grouped_results = {}
        for result in results:
            if result["source"] == "discourse":
                key = f"discourse_{result['post_id']}"
            else:
                key = f"markdown_{result['title']}"
            
            if key not in grouped_results:
                grouped_results[key] = []
            grouped_results[key].append(result)
        
        # Keep top chunks per source
        final_results = []
        for key, chunks in grouped_results.items():
            chunks.sort(key=lambda x: x["similarity"], reverse=True)
            final_results.extend(chunks[:MAX_CONTEXT_CHUNKS])
        
        final_results.sort(key=lambda x: x["similarity"], reverse=True)
        return final_results[:MAX_RESULTS]
        
    except Exception as e:
        error_msg = f"Error in find_similar_content: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise

# Function to generate answer using LLM
async def generate_answer(question, relevant_results, max_retries=2):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API_KEY not configured. Please set the API_KEY environment variable.")
    
    retries = 0
    while retries < max_retries:    
        try:
            logger.info(f"Generating answer for question: '{question[:50]}...'")
            context = ""
            for result in relevant_results:
                source_type = "Discourse post" if result["source"] == "discourse" else "Documentation"
                context += f"\n\n{source_type} (URL: {result['url']}):\n{result['content'][:1500]}"
            
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
                async with session.post(url, headers=headers, json=payload, timeout=60) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info("Successfully received answer from LLM")
                        return result["choices"][0]["message"]["content"]
                    elif response.status == 429:
                        error_text = await response.text()
                        logger.warning(f"Rate limit reached, retrying after delay (retry {retries+1}): {error_text}")
                        await asyncio.sleep(3 * (retries + 1))
                        retries += 1
                    else:
                        error_text = await response.text()
                        error_msg = f"Error generating answer (status {response.status}): {error_text}"
                        logger.error(error_msg)
                        raise HTTPException(status_code=response.status, detail=error_msg)
        except asyncio.TimeoutError:
            logger.warning(f"Timeout generating answer (retry {retries+1})")
            retries += 1
            if retries >= max_retries:
                raise HTTPException(status_code=500, detail="Timeout generating answer")
            await asyncio.sleep(2)
        except Exception as e:
            error_msg = f"Exception generating answer: {e}"
            logger.error(error_msg)
            retries += 1
            if retries >= max_retries:
                raise HTTPException(status_code=500, detail=error_msg)
            await asyncio.sleep(2)

# Function to process multimodal content
async def process_multimodal_query(question, image_base64):
    try:
        logger.info(f"Processing query: '{question[:50]}...', image provided: {image_base64 is not None}")
        if not image_base64:
            return await get_embedding(question)
        
        if not API_KEY:
            raise HTTPException(status_code=500, detail="API_KEY not configured. Please set the API_KEY environment variable.")
        
        # Process image with GPT-4o Vision
        url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        headers = {
            "Authorization": API_KEY,
            "Content-Type": "application/json"
        }
        
        image_content = f"data:image/jpeg;base64,{image_base64}"
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Look at this image and tell me what you see related to this question: {question}"},
                        {"type": "image_url", "image_url": {"url": image_content}}
                    ]
                }
            ]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=60) as response:
                if response.status == 200:
                    result = await response.json()
                    image_description = result["choices"][0]["message"]["content"]
                    combined_query = f"{question}\nImage context: {image_description}"
                    return await get_embedding(combined_query)
                else:
                    logger.error(f"Error processing image: {response.status}")
                    return await get_embedding(question)
    except Exception as e:
        logger.error(f"Exception processing multimodal query: {e}")
        return await get_embedding(question)

# Parse LLM response
def parse_llm_response(response):
    try:
        parts = response.split("Sources:", 1)
        
        if len(parts) == 1:
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
                    
                line = re.sub(r'^\d+\.\s*', '', line)
                line = re.sub(r'^-\s*', '', line)
                
                url_match = re.search(r'URL:\s*\[(.*?)\]|url:\s*\[(.*?)\]|\[(http[^\]]+)\]|URL:\s*(http\S+)|url:\s*(http\S+)|(http\S+)', line, re.IGNORECASE)
                text_match = re.search(r'Text:\s*\[(.*?)\]|text:\s*\[(.*?)\]|[""](.*?)[""]|Text:\s*"(.*?)"|text:\s*"(.*?)"', line, re.IGNORECASE)
                
                if url_match:
                    url = next((g for g in url_match.groups() if g), "").strip()
                    text = "Source reference"
                    
                    if text_match:
                        text_value = next((g for g in text_match.groups() if g), "")
                        if text_value:
                            text = text_value.strip()
                    
                    if url and url.startswith("http"):
                        links.append({"url": url, "text": text})
        
        return {"answer": answer, "links": links}
    except Exception as e:
        logger.error(f"Error parsing LLM response: {e}")
        return {
            "answer": "Error parsing the response from the language model.",
            "links": []
        }

# API Routes
@app.get("/")
async def root():
    # Check configuration status
    config_status = {
        "api_key_configured": API_KEY is not None,
        "environment": "vercel" if os.getenv('VERCEL') else "local",
        "database_available": not os.getenv('VERCEL')  # Database only available locally
    }
    
    status_message = ""
    if not config_status["api_key_configured"]:
        status_message += "⚠️ API_KEY not configured. "
    if config_status["environment"] == "vercel" and not config_status["database_available"]:
        status_message += "⚠️ Database not configured for production. "
    
    if not status_message:
        status_message = "✅ All systems configured correctly."
    
    return HTMLResponse(content=f"""
    <html>
        <head>
            <title>RAG Query API</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 2em; }}
                .status {{ padding: 1em; margin: 1em 0; border-radius: 5px; }}
                .warning {{ background-color: #fff3cd; border: 1px solid #ffeaa7; }}
                .success {{ background-color: #d4edda; border: 1px solid #c3e6cb; }}
            </style>
        </head>
        <body>
            <h1>RAG Query API</h1>
            <div class="status {'success' if not status_message.startswith('⚠️') else 'warning'}">
                <strong>Status:</strong> {status_message}
            </div>
            <p>API is running. Use POST /api/ to make queries.</p>
            <p>Environment: {config_status["environment"]}</p>
            <p>API Key: {'✅ Configured' if config_status["api_key_configured"] else '❌ Not configured'}</p>
            <p>Database: {'✅ Available' if config_status["database_available"] else '❌ Not configured'}</p>
            <p>For the full interface, serve the index.html file separately.</p>
        </body>
    </html>
    """, status_code=200)

@app.post("/api/", response_model=QueryResponse)
async def api_query(request: Request):
    try:
        logger.info(f"Received request from: {request.client.host if request.client else 'unknown'}")
        
        body = await request.json()
        question = body.get("question")
        image = body.get("image")
        
        if not question:
            return JSONResponse(
                status_code=400,
                content={"answer": "Error: Question is required", "links": []}
            )
        
        logger.info(f"Received API query: '{question[:50]}...'")
        
        # Check if API key is configured
        if not API_KEY:
            return JSONResponse(
                status_code=500,
                content={
                    "answer": "API_KEY not configured. Please set the API_KEY environment variable in your Vercel dashboard.",
                    "links": []
                }
            )
        
        # Check if we're in Vercel and database isn't set up
        if os.getenv('VERCEL'):
            return JSONResponse(
                status_code=500,
                content={
                    "answer": "Database not configured for production deployment. Please set up Vercel Postgres or another cloud database service.",
                    "links": []
                }
            )
        
        conn = get_db_connection()
        try:
            query_embedding = await process_multimodal_query(question, image)
            similar_results = await find_similar_content(query_embedding, conn)
            
            if not similar_results:
                return {
                    "answer": "I don't have enough information to answer this question based on the available knowledge base.", 
                    "links": []
                }
            
            llm_response = await generate_answer(question, similar_results)
            parsed_response = parse_llm_response(llm_response)
            
            return QueryResponse(
                answer=parsed_response["answer"],
                links=[LinkInfo(**link) for link in parsed_response["links"]]
            )
        finally:
            conn.close()
            
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "answer": f"Server error: {str(e)}",
                "links": []
            }
        )

@app.get("/health")
async def health_check():
    try:
        health_status = {
            "status": "healthy",
            "environment": "vercel" if os.getenv('VERCEL') else "local",
            "api_key_configured": API_KEY is not None,
            "database": "not_configured" if os.getenv('VERCEL') else "connected"
        }
        
        if not os.getenv('VERCEL'):
            # Test database connection only if not on Vercel
            conn = get_db_connection()
            conn.close()
            health_status["database"] = "connected"
        
        return health_status
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=500, 
            content={
                "status": "unhealthy", 
                "error": str(e),
                "api_key_configured": API_KEY is not None
            }
        )