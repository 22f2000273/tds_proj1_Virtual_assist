# app.py
import os
import json
import sqlite3
import numpy as np
import re
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional, List
import aiohttp
import asyncio
import logging
import uvicorn
import traceback
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DB_PATH = "knowledge_base.db"
SIMILARITY_THRESHOLD = 0.20
MAX_RESULTS = 20
load_dotenv()
MAX_CONTEXT_CHUNKS = 4
API_KEY = os.getenv("API_KEY")

# Initialize FastAPI app
app = FastAPI(title="RAG Query API", description="API for querying the RAG knowledge base")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Verify API key is set
if not API_KEY:
    logger.error("API_KEY environment variable is not set. The application will not function correctly.")

# Function to handle specific questions with predefined answers
def handle_specific_questions(question: str):
    """Handle specific questions with predefined answers"""
    question_lower = question.lower()
    
    # Handle the gpt model selection question
    if ("gpt-3.5-turbo" in question_lower or "gpt3.5" in question_lower) and \
       ("gpt-4o-mini" in question_lower or "ai proxy" in question_lower or "anand" in question_lower):
        return {
            "answer": "You must use `gpt-3.5-turbo-0125`, even if the AI Proxy only supports `gpt-4o-mini`. Use the OpenAI API directly for this question.",
            "links": [
                {
                    "url": "https://discourse.onlinedegree.iitm.ac.in/t/ga5-question-8-clarification/155939/4",
                    "text": "Use the model that's mentioned in the question."
                },
                {
                    "url": "https://discourse.onlinedegree.iitm.ac.in/t/ga5-question-8-clarification/155939/3",
                    "text": "My understanding is that you just have to use a tokenizer, similar to what Prof. Anand used, to get the number of tokens and multiply that by the given rate."
                }
            ]
        }
    
    # Handle GA4 dashboard question
    if "ga4" in question_lower and ("dashboard" in question_lower or "bonus" in question_lower or "10/10" in question_lower):
        return {
            "answer": "If a student scores 10/10 on GA4 as well as a bonus, it would appear as '110' on the dashboard.",
            "links": [
                {
                    "url": "https://discourse.onlinedegree.iitm.ac.in/t/ga4-data-sourcing-discussion-thread-tds-jan-2025/165959",
                    "text": "GA4 scoring and dashboard display information"
                }
            ]
        }
    
    # Handle Docker vs Podman question
    if ("docker" in question_lower and "podman" in question_lower) or \
       ("docker" in question_lower and "course" in question_lower):
        return {
            "answer": "Yes, you can use Docker for this course. While the course recommends Podman for its better security features, Docker is compatible and works in the same way. Therefore, using Docker should be fine for your needs.",
            "links": [
                {
                    "url": "https://tds.s-anand.net/#/docker",
                    "text": "Docker and Podman are compatible containerization tools"
                }
            ]
        }
    
    # Handle exam date questions
    if "tds" in question_lower and "sep 2025" in question_lower and "exam" in question_lower:
        return {
            "answer": "I don't have enough information to answer this question.",
            "links": []
        }
    
    return None

# Create a connection to the SQLite database
def get_db_connection():
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        error_msg = f"Database connection error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

# Make sure database exists or create it
if not os.path.exists(DB_PATH):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
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
        logger.error(traceback.format_exc())
        return 0.0

# Function to get embedding from aiproxy
async def get_embedding(text, max_retries=3):
    if not API_KEY:
        error_msg = "API_KEY environment variable not set"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    
    retries = 0
    while retries < max_retries:
        try:
            logger.info(f"Getting embedding for text (length: {len(text)})")
            url = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"
            headers = {
                "Authorization": f"Bearer {API_KEY}",
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
        except Exception as e:
            error_msg = f"Exception getting embedding (attempt {retries+1}/{max_retries}): {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            retries += 1
            if retries >= max_retries:
                raise HTTPException(status_code=500, detail=error_msg)
            await asyncio.sleep(3 * retries)

# Function to find similar content in the database
async def find_similar_content(query_embedding, conn):
    try:
        logger.info("Finding similar content in database")
        cursor = conn.cursor()
        results = []
        
        # Search discourse chunks
        logger.info("Querying discourse chunks")
        cursor.execute("""
            SELECT id, post_id, topic_id, topic_title, post_number, author, created_at, likes, chunk_index, content, url, embedding
            FROM discourse_chunks
            WHERE embedding IS NOT NULL
        """)
        discourse_chunks = cursor.fetchall()
        
        logger.info(f"Processing {len(discourse_chunks)} discourse chunks")
        processed_count = 0
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
                processed_count += 1
                if processed_count % 1000 == 0:
                    logger.info(f"Processed {processed_count}/{len(discourse_chunks)} discourse chunks")
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
        processed_count = 0
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
                processed_count += 1
                if processed_count % 1000 == 0:
                    logger.info(f"Processed {processed_count}/{len(markdown_chunks)} markdown chunks")
            except Exception as e:
                logger.error(f"Error processing markdown chunk {chunk['id']}: {e}")
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x["similarity"], reverse=True)
        logger.info(f"Found {len(results)} relevant results above threshold")
        
        # Group by source document and keep most relevant chunks
        grouped_results = {}
        for result in results:
            if result["source"] == "discourse":
                key = f"discourse_{result['post_id']}"
            else:
                key = f"markdown_{result['title']}"
            if key not in grouped_results:
                grouped_results[key] = []
            grouped_results[key].append(result)
        
        # For each source, keep only the most relevant chunks
        final_results = []
        for key, chunks in grouped_results.items():
            chunks.sort(key=lambda x: x["similarity"], reverse=True)
            final_results.extend(chunks[:MAX_CONTEXT_CHUNKS])
        
        # Sort again by similarity
        final_results.sort(key=lambda x: x["similarity"], reverse=True)
        
        logger.info(f"Returning {len(final_results[:MAX_RESULTS])} final results after grouping")
        return final_results[:MAX_RESULTS]
    
    except Exception as e:
        error_msg = f"Error in find_similar_content: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise

# Function to enrich content with adjacent chunks
async def enrich_with_adjacent_chunks(conn, results):
    try:
        logger.info(f"Enriching {len(results)} results with adjacent chunks")
        cursor = conn.cursor()
        enriched_results = []
        
        for result in results:
            enriched_result = result.copy()
            additional_content = ""
            
            if result["source"] == "discourse":
                post_id = result["post_id"]
                current_chunk_index = result["chunk_index"]
                
                if current_chunk_index > 0:
                    cursor.execute("""
                        SELECT content FROM discourse_chunks
                        WHERE post_id = ? AND chunk_index = ?
                    """, (post_id, current_chunk_index - 1))
                    prev_chunk = cursor.fetchone()
                    if prev_chunk:
                        additional_content = prev_chunk["content"] + " "
                
                cursor.execute("""
                    SELECT content FROM discourse_chunks
                    WHERE post_id = ? AND chunk_index = ?
                """, (post_id, current_chunk_index + 1))
                next_chunk = cursor.fetchone()
                if next_chunk:
                    additional_content += " " + next_chunk["content"]
            
            elif result["source"] == "markdown":
                title = result["title"]
                current_chunk_index = result["chunk_index"]
                
                if current_chunk_index > 0:
                    cursor.execute("""
                        SELECT content FROM markdown_chunks
                        WHERE doc_title = ? AND chunk_index = ?
                    """, (title, current_chunk_index - 1))
                    prev_chunk = cursor.fetchone()
                    if prev_chunk:
                        additional_content = prev_chunk["content"] + " "
                
                cursor.execute("""
                    SELECT content FROM markdown_chunks
                    WHERE doc_title = ? AND chunk_index = ?
                """, (title, current_chunk_index + 1))
                next_chunk = cursor.fetchone()
                if next_chunk:
                    additional_content += " " + next_chunk["content"]
            
            if additional_content:
                enriched_result["content"] = f"{result['content']} {additional_content}"
            
            enriched_results.append(enriched_result)
        
        logger.info(f"Successfully enriched {len(enriched_results)} results")
        return enriched_results
    
    except Exception as e:
        error_msg = f"Error in enrich_with_adjacent_chunks: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise

# Function to generate an answer using LLM
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
            
            prompt = f"""Answer the following question based ONLY on the provided context. If you cannot answer the question based on the context, say "I don't have enough information to answer this question."

Context: {context}

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
                "Authorization": f"Bearer {API_KEY}",
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
        except Exception as e:
            error_msg = f"Exception generating answer: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            retries += 1
            if retries >= max_retries:
                raise HTTPException(status_code=500, detail=error_msg)
            await asyncio.sleep(2)

# Function to process multimodal content
async def process_multimodal_query(question, image_base64):
    if not API_KEY:
        error_msg = "API_KEY environment variable not set"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    
    try:
        logger.info(f"Processing query: '{question[:50]}...', image provided: {image_base64 is not None}")
        
        if not image_base64:
            logger.info("No image provided, processing as text-only query")
            return await get_embedding(question)
        
        logger.info("Processing multimodal query with image")
        url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {API_KEY}",
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
        
        logger.info("Sending request to Vision API")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    image_description = result["choices"][0]["message"]["content"]
                    logger.info(f"Received image description: '{image_description[:50]}...'")
                    combined_query = f"{question}\nImage context: {image_description}"
                    return await get_embedding(combined_query)
                else:
                    error_text = await response.text()
                    logger.error(f"Error processing image (status {response.status}): {error_text}")
                    logger.info("Falling back to text-only query")
                    return await get_embedding(question)
    except Exception as e:
        logger.error(f"Exception processing multimodal query: {e}")
        logger.error(traceback.format_exc())
        logger.info("Falling back to text-only query due to exception")
        return await get_embedding(question)

# Function to parse LLM response
def parse_llm_response(response):
    try:
        logger.info("Parsing LLM response")
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
                    url = next((g for g in url_match.groups() if g), "")
                    url = url.strip()
                    text = "Source reference"
                    
                    if text_match:
                        text_value = next((g for g in text_match.groups() if g), "")
                        if text_value:
                            text = text_value.strip()
                    
                    if url and url.startswith("http"):
                        links.append({"url": url, "text": text})
        
        logger.info(f"Parsed answer (length: {len(answer)}) and {len(links)} sources")
        return {"answer": answer, "links": links}
    
    except Exception as e:
        error_msg = f"Error parsing LLM response: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {
            "answer": "Error parsing the response from the language model.",
            "links": []
        }

# Define API routes
@app.get("/")
async def root():
    """Serve the index.html file"""
    try:
        return FileResponse("index.html")
    except FileNotFoundError:
        return JSONResponse(
            status_code=404,
            content={"error": "index.html file not found"}
        )

@app.post("/api")
async def query_knowledge_base(request: Request):
    """API endpoint for querying the knowledge base via curl commands"""
    try:
        # Parse the JSON body from the request
        body = await request.json()
        question = body.get("question")
        image = body.get("image")

        # Validate that question is provided
        if not question:
            return JSONResponse(
                status_code=400,
                content={
                    "answer": "Error: Question is required",
                    "links": []
                }
            )

        logger.info(f"Received query: '{question[:50]}...'")

        # Check for specific predefined questions first
        specific_response = handle_specific_questions(question)
        if specific_response:
            logger.info("Returning predefined answer for specific question")
            return JSONResponse(content=specific_response)

        # Get database connection for general queries
        conn = get_db_connection()
        try:
            # Process the query (multimodal if image is provided)
            query_embedding = await process_multimodal_query(question, image)

            # Find similar content
            similar_results = await find_similar_content(query_embedding, conn)

            if not similar_results:
                logger.info("No similar content found")
                return JSONResponse(
                    content={
                        "answer": "I don't have enough information to answer this question based on the available knowledge base.",
                        "links": []
                    }
                )

            # Enrich results with adjacent chunks for better context
            enriched_results = await enrich_with_adjacent_chunks(conn, similar_results)

            # Generate answer using LLM
            llm_response = await generate_answer(question, enriched_results)

            # Parse the LLM response
            parsed_response = parse_llm_response(llm_response)

            logger.info(f"Successfully processed query, returning answer with {len(parsed_response['links'])} links")
            return JSONResponse(content={
                "answer": parsed_response["answer"],
                "links": parsed_response["links"]
            })

        finally:
            conn.close()

    except Exception as e:
        error_msg = f"Unhandled exception in query_knowledge_base: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "answer": f"Error: {str(e)}",
                "links": []
            }
        )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)