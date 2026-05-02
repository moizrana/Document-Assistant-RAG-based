import os
import time
import uuid
import logging
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import json

from src.config import settings
from src.document_loader import load_document
from src.chunker import chunk_documents
from src.embedder import embed_texts, get_embedding_dimension
from src.session_store import RedisSessionStore, InMemorySessionStore, UserSession
from src.pipeline import run_rag_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DocuMind AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()],  # default: "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================
# Session Management (Redis-backed with local hot cache)
# =====================================================================
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_SESSION_TTL_SECONDS = int(os.getenv("REDIS_SESSION_TTL_SECONDS", str(60 * 60 * 24)))

_session_store = None


@app.on_event("startup")
async def _startup():
    global _session_store
    from redis.asyncio import Redis

    try:
        redis_client = Redis.from_url(REDIS_URL, decode_responses=False)
        await redis_client.ping()
        _session_store = RedisSessionStore(redis_client, ttl_seconds=REDIS_SESSION_TTL_SECONDS)
        logger.info("Redis session store initialized")
    except Exception as e:
        _session_store = InMemorySessionStore()
        logger.warning(f"Redis unavailable, using in-memory sessions. ({type(e).__name__}: {e})")


def _require_store():
    if _session_store is None:
        raise RuntimeError("Session store not initialized")
    return _session_store


async def get_session(session_id: str) -> UserSession:
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id header is required")
    store = _require_store()
    return await store.get(session_id)

# Periodically clean up old sessions (optional, for production)
# def cleanup_sessions():
#    ...

# =====================================================================
# API Models
# =====================================================================
class AskRequest(BaseModel):
    query: str
    session_id: str
    use_query_expansion: bool = False
    use_evaluation: bool = True
    use_cache: bool = True

class ClearSessionRequest(BaseModel):
    session_id: str

# =====================================================================
# Endpoints
# =====================================================================
@app.post("/api/upload")
async def upload_documents(
    session_id: str = Form(...),
    files: List[UploadFile] = File(...)
):
    session = await get_session(session_id)
    all_chunks = []
    total_docs = 0

    for uploaded_file in files:
        if uploaded_file.filename in session.uploaded_files:
            continue
        
        file_bytes = await uploaded_file.read()
        try:
            documents = load_document(file_bytes, uploaded_file.filename)
            total_docs += len(documents)
            
            chunks = chunk_documents(
                documents,
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP
            )
            all_chunks.extend(chunks)
            session.uploaded_files.append(uploaded_file.filename)
        except Exception as e:
            logger.error(f"Error processing {uploaded_file.filename}: {e}")
            raise HTTPException(status_code=400, detail=f"Error processing {uploaded_file.filename}")

    if not all_chunks:
        return {"message": "No new content extracted.", "chunks_processed": 0}

    # Embed chunks
    texts = [chunk.content for chunk in all_chunks]
    embeddings = embed_texts(texts, settings.EMBEDDING_MODEL)

    # Update Indices
    session.faiss_index.add(embeddings, all_chunks)
    session.bm25_index.add(all_chunks)
    session.documents_loaded += total_docs

    # Persist updated session to Redis (upload is the state-changing operation)
    await _require_store().save(session_id)

    return {
        "message": "Documents processed successfully",
        "documents_loaded": total_docs,
        "chunks_indexed": len(all_chunks),
        "total_files": session.uploaded_files
    }


@app.post("/api/ask")
async def ask_question(request: AskRequest):
    session = await get_session(request.session_id)

    # Check for simple greetings
    cleaned_prompt = request.query.strip().lower()
    is_small_talk = False
    if cleaned_prompt.rstrip("!?.,") in ["hey", "hello", "hi", "how are you", "good morning", "good evening"]:
        is_small_talk = True
    if "who are you" in cleaned_prompt or "your name" in cleaned_prompt:
        is_small_talk = True

    if is_small_talk:
        final_response = {
            "answer": "Hello! I am DocuMind AI, your Intelligent Document Assistant. How can I help you today?",
            "sources": [],
            "model_used": "system",
            "eval_scores": None
        }
        async def event_generator():
            yield f"data: {json.dumps({'chunk': final_response['answer']})}\n\n"
            yield f"data: {json.dumps({'final': final_response})}\n\n"
            yield "event: done\ndata: \n\n"
        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # Real RAG Execution with Streaming
    import queue
    import threading

    # We use a thread-safe queue to pass tokens from the generator thread to the async streamer
    q = queue.Queue()

    def stream_callback(token):
        q.put({"type": "chunk", "content": token})

    def run_pipeline():
        try:
            result = run_rag_pipeline(
                query=request.query,
                faiss_index=session.faiss_index,
                bm25_index=session.bm25_index,
                semantic_cache=session.semantic_cache,
                use_guardrails=True,
                use_cache=request.use_cache,
                use_query_expansion=request.use_query_expansion,
                use_evaluation=request.use_evaluation,
                stream_callback=stream_callback
            )
            q.put({"type": "final", "content": result})
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            q.put({"type": "error", "content": str(e)})

    # Start pipeline in a background thread so we can stream from the queue asynchronously
    thread = threading.Thread(target=run_pipeline)
    thread.start()

    async def event_generator():
        while True:
            try:
                # Non-blocking get with sleep allows asyncio to yield control
                item = q.get_nowait()
                if item["type"] == "chunk":
                    yield f"data: {json.dumps({'chunk': item['content']})}\n\n"
                elif item["type"] == "final":
                    yield f"data: {json.dumps({'final': item['content']})}\n\n"
                    break
                elif item["type"] == "error":
                    yield f"data: {json.dumps({'error': item['content']})}\n\n"
                    break
            except queue.Empty:
                await asyncio.sleep(0.05)
        
        yield "event: done\ndata: \n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.delete("/api/session")
async def clear_session(request: ClearSessionRequest):
    await _require_store().delete(request.session_id)
    return {"message": "Session cleared"}
