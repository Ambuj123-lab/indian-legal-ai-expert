"""
RAG API Routes — Chat, Streaming, Admin, Feedback
"""

import asyncio
import json
import logging
import time
from datetime import datetime

from fastapi import APIRouter, Request, Depends, HTTPException, UploadFile, File, Body

from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List

from app.auth.jwt import get_current_user, get_admin_user
from app.config import get_settings
from app.db.database import get_chat_history, save_feedback, save_message
from app.rag.graph import get_rag_graph
from app.rag.pipeline import (
    search_similar, generate_response_stream, mask_pii,
    is_abusive, is_greeting, sync_knowledge_base,
    index_temp_file, get_user_temp_files, delete_user_temp_file
)
from app.db.supabase_client import get_registry_stats, get_all_registry_entries

router = APIRouter(prefix="/api", tags=["RAG"])
settings = get_settings()
logger = logging.getLogger(__name__)


# --- Redis helper (safe mode) ---
def get_redis():
    try:
        from upstash_redis import Redis
        if settings.UPSTASH_REDIS_REST_URL:
            r = Redis(url=settings.UPSTASH_REDIS_REST_URL, token=settings.UPSTASH_REDIS_REST_TOKEN)
            logger.info("✅ Redis connected successfully")
            return r
        else:
            logger.warning("⚠️ UPSTASH_REDIS_REST_URL not set in .env")
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
    return None


# --- Pydantic Models ---

class ChatRequest(BaseModel):
    question: str
    use_streaming: bool = False


class FeedbackRequest(BaseModel):
    question: str
    response: str
    rating: str  # "up" or "down"


from app.limiter import limiter  # Rate limiter

@router.post("/chat")
@limiter.limit("5/minute")
async def chat(request: Request, body: ChatRequest = Body(...), user: dict = Depends(get_current_user)):
    """
    Main chat endpoint — uses LangGraph pipeline.
    Supports optional Redis caching.
    """
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    user_email = user.get("email", "anonymous")
    user_name = user.get("name", "User")

    # Redis cache check
    redis = get_redis()
    cache_key = f"chat:{user_email}:{question[:100]}"
    if redis:
        try:
            cached = redis.get(cache_key)
            if cached:
                logger.info("Cache hit")
                return json.loads(cached)
        except Exception:
            pass

    # Track active user
    if redis:
        try:
            redis.setex(f"active:{user_email}", 900, "1")  # 15 min sliding window
        except Exception:
            pass

    # Get chat history
    history = get_chat_history(user_email, limit=6)

    # Run LangGraph pipeline
    graph = get_rag_graph()
    result = graph.invoke({
        "query": question,
        "user_name": user_name,
        "user_email": user_email,
        "chat_history": history,
        "query_type": "",
        "safe_query": "",
        "pii_found": False,
        "pii_entities": [],
        "context": "",
        "sources": [],
        "confidence": 0.0,
        "response": "",
        "latency": 0.0,
        "error": None
    })

    response_data = {
        "response": result.get("response", "No response generated"),
        "sources": result.get("sources", []),
        "confidence": result.get("confidence", 0),
        "latency": result.get("latency", 0),
        "pii_detected": result.get("pii_found", False),
        "query_type": result.get("query_type", "unknown"),
        "timestamp": datetime.now().isoformat()
    }

    # Cache response (1 hour TTL)
    if redis:
        try:
            redis.setex(cache_key, 3600, json.dumps(response_data))
        except Exception:
            pass

    return response_data

@router.post("/chat/stream")
@limiter.limit("5/minute")
async def chat_stream(request: Request, body: ChatRequest = Body(...), user: dict = Depends(get_current_user)):
    """
    Streaming chat endpoint — returns tokens via Server-Sent Events.
    Same LLM call, same tokens. Just delivery method changes.
    """
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    user_email = user.get("email", "anonymous")
    user_name = user.get("name", "User")

    # Track active user
    redis = get_redis()
    if redis:
        try:
            redis.setex(f"active:{user_email}", 900, "1")
            logger.info(f"✅ Active user tracked: {user_email}")
        except Exception as e:
            logger.error(f"❌ Redis active user tracking failed: {e}")

    # Redis cache check — same question = instant answer
    cache_key = f"stream:{user_email}:{question[:100]}"
    if redis:
        try:
            cached = redis.get(cache_key)
            if cached:
                logger.info(f"⚡ Cache HIT for: {question[:50]}")
                cached_data = json.loads(cached)
                async def cached_stream():
                    # Simulate streaming — send word by word (fast)
                    words = cached_data['response'].split(' ')
                    for i, word in enumerate(words):
                        token = word if i == 0 else ' ' + word
                        yield f"data: {json.dumps({'token': token})}\n\n"
                        await asyncio.sleep(0.02)  # 20ms per word = fast but visible
                    yield f"data: {json.dumps({'done': True, 'sources': cached_data.get('sources', []), 'confidence': cached_data.get('confidence', 0), 'pii_detected': cached_data.get('pii_detected', False), 'pii_entities': cached_data.get('pii_entities', [])})}\n\n"
                return StreamingResponse(cached_stream(), media_type="text/event-stream")
        except Exception as e:
            logger.error(f"❌ Redis cache check failed: {e}")

    # Classify first
    if is_abusive(question):
        async def abort_stream():
            yield f"data: {json.dumps({'token': 'I am a Legal AI Assistant. I can only respond to professional and respectful queries.'})}\n\n"
            yield f"data: {json.dumps({'done': True, 'sources': [], 'confidence': 0})}\n\n"
        return StreamingResponse(abort_stream(), media_type="text/event-stream")

    # PII mask
    safe_query, pii_found, pii_entities = mask_pii(question)

    # Check for greeting
    if is_greeting(question):
        async def greet_stream():
            yield f"data: {json.dumps({'token': f'Hello {user_name}! 👋 I am Indian Legal AI Expert. Ask me about Constitution, BNS, Consumer Protection, IT Act, and more!'})}\n\n"
            yield f"data: {json.dumps({'done': True, 'sources': [], 'confidence': 100})}\n\n"

        # Save greeting to history
        save_message(user_email, "user", question)
        save_message(user_email, "assistant", f"Hello {user_name}! 👋 I am Indian Legal AI Expert.")
        return StreamingResponse(greet_stream(), media_type="text/event-stream")

    # Retrieve context (searches BOTH core + user's temp files)
    results = search_similar(safe_query, top_k=5, user_email=user_email)
    confidence = results[0]["score"] * 100 if results else 0

    sources = [
        {
            "source_id": i + 1,
            "file": r["source_file"].replace(".pdf", ""),
            "page": r["page"] + 1,
            "preview": r["child_text"][:300],
            "score": round(r["score"], 3)
        }
        for i, r in enumerate(results)
    ] if results else []

    context = "\n\n---\n\n".join([r["parent_text"] for r in results]) if results else ""

    # Get chat history
    history = get_chat_history(user_email, limit=6)
    history_text = "No previous history."
    if history:
        msgs = history[-6:]
        formatted = [
            ("User: " if m.get("role") == "user" else "Assistant: ") + str(m.get("content", ""))
            for m in msgs
        ]
        history_text = "\n".join(formatted)

    # Low confidence fallback
    if confidence < 40:
        async def fallback_stream():
            msg = "I don't have sufficient information in my knowledge base to answer this accurately based on the provided legal documents."
            yield f"data: {json.dumps({'token': msg})}\n\n"
            yield f"data: {json.dumps({'done': True, 'sources': [], 'confidence': round(confidence, 1)})}\n\n"
        save_message(user_email, "user", safe_query, pii_masked=pii_found, pii_entities=pii_entities)
        return StreamingResponse(fallback_stream(), media_type="text/event-stream")

    # Stream LLM response
    full_response = []

    async def sse_stream():
        async for token in generate_response_stream(
            question=safe_query,
            context=context,
            history=history_text,
            user_name=user_name
        ):
            full_response.append(token)
            yield f"data: {json.dumps({'token': token})}\n\n"

        # Final event with metadata
        yield f"data: {json.dumps({'done': True, 'sources': sources, 'confidence': round(confidence, 1), 'pii_detected': pii_found, 'pii_entities': pii_entities})}\n\n"

        # Save to MongoDB after streaming completes
        complete_response = "".join(full_response)
        save_message(user_email, "user", safe_query, pii_masked=pii_found, pii_entities=pii_entities)
        save_message(user_email, "assistant", complete_response, sources)

        # Cache response in Redis (1 hour TTL)
        if redis:
            try:
                cache_data = json.dumps({
                    "response": complete_response,
                    "sources": sources,
                    "confidence": round(confidence, 1),
                    "pii_detected": pii_found,
                    "pii_entities": pii_entities
                })
                redis.setex(cache_key, 3600, cache_data)
                logger.info(f"💾 Response cached for: {question[:50]}")
            except Exception as e:
                logger.error(f"❌ Redis cache save failed: {e}")

    return StreamingResponse(sse_stream(), media_type="text/event-stream")


# ========================
# FEEDBACK
# ========================

@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest, user: dict = Depends(get_current_user)):
    """Submit feedback (👍/👎) for a response"""
    save_feedback(user["email"], request.question, request.response, request.rating)
    return {"message": "Feedback recorded", "rating": request.rating}


# ========================
# CHAT HISTORY
# ========================

@router.get("/history")
async def get_history(user: dict = Depends(get_current_user)):
    """Get chat history for current user"""
    history = get_chat_history(user["email"], limit=50)
    return {"history": history}


@router.delete("/history")
async def clear_history(user: dict = Depends(get_current_user)):
    """Clear chat history"""
    from app.db.database import get_chat_collection
    collection = get_chat_collection()
    if collection:
        collection.delete_one({"user_email": user["email"]})
    return {"message": "History cleared"}


# ========================
# USER INFO
# ========================

@router.get("/me")
async def get_me(user: dict = Depends(get_current_user)):
    """Return current user info + admin flag"""
    return {
        "email": user["email"],
        "name": user["name"],
        "picture": user["picture"],
        "is_admin": user["email"].lower() == settings.ADMIN_EMAIL.lower()
    }


# ========================
# ADMIN ENDPOINTS (admin-only)
# ========================

@router.post("/admin/documents/sync")
async def sync_documents(user: dict = Depends(get_admin_user)):
    """
    Sync knowledge base:
    - Compare Supabase Storage files with document_registry
    - Auto: add new, update changed, delete orphans
    """
    try:
        results = sync_knowledge_base()
        return {
            "message": "Sync complete",
            "results": {
                "added": results["added"],
                "updated": results["updated"],
                "deleted": results["deleted"],
                "unchanged": results["unchanged"],
                "errors": results["errors"],
                "summary": {
                    "added_count": len(results["added"]),
                    "updated_count": len(results["updated"]),
                    "deleted_count": len(results["deleted"]),
                    "unchanged_count": len(results["unchanged"]),
                    "error_count": len(results["errors"])
                }
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Sync error: {e}")
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")


@router.get("/admin/documents")
async def list_documents(user: dict = Depends(get_current_user)):
    """List all documents with their status"""
    stats = get_registry_stats()
    return stats


@router.delete("/admin/documents/{file_name}")
async def delete_document(file_name: str, user: dict = Depends(get_admin_user)):
    """Delete a document — removes from Qdrant + marks deleted in registry"""
    from app.rag.pipeline import delete_file_from_qdrant
    from app.db.supabase_client import mark_registry_deleted, delete_storage_file

    try:
        delete_file_from_qdrant(file_name)
        delete_storage_file(file_name)
        mark_registry_deleted(file_name)
        return {"message": f"Deleted: {file_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========================
# USER TEMP UPLOADS
# ========================

@router.post("/upload")
async def upload_temp_file(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    """
    Upload a temporary PDF — indexed for this user's session only.
    On logout, ALL temp vectors are deleted. Core brain UNTOUCHED.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    if file.size and file.size > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")

    user_email = user.get("email", "anonymous")
    file_bytes = await file.read()

    try:
        stats = index_temp_file(file.filename, file_bytes, user_email)
        return {
            "message": f"Uploaded and indexed: {file.filename}",
            "file_name": file.filename,
            "parent_chunks": stats["parent_count"],
            "child_chunks": stats["child_count"],
            "is_temporary": True,
            "note": "This file will be automatically removed when you logout."
        }
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/uploads")
async def list_temp_uploads(user: dict = Depends(get_current_user)):
    """List all temporarily uploaded files for this user"""
    files = get_user_temp_files(user["email"])
    return {"files": files, "count": len(files)}


@router.delete("/uploads/{file_name}")
async def delete_temp_upload(file_name: str, user: dict = Depends(get_current_user)):
    """Delete a specific temp upload (without touching core brain)"""
    success = delete_user_temp_file(user["email"], file_name)
    if success:
        return {"message": f"Deleted temp file: {file_name}"}
    raise HTTPException(status_code=500, detail="Delete failed")


# ========================
# STATS
# ========================

@router.get("/stats")
async def get_stats():
    """Public stats endpoint — visitor count + KB status"""
    redis = get_redis()
    active_users = 0

    if redis:
        try:
            keys = redis.keys("active:*")
            active_users = len(keys) if keys else 0
        except Exception:
            pass

    stats = get_registry_stats()
    return {
        "active_users": active_users,
        "total_documents": stats.get("total_documents", 0),
        "total_chunks": stats.get("total_chunks", 0),
        "total_parent_chunks": stats.get("total_parent_chunks", 0),
        "total_child_chunks": stats.get("total_child_chunks", 0),
        "timestamp": datetime.now().isoformat()
    }
