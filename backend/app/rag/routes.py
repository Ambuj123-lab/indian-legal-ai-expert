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
from app.core.config import get_settings
from app.db.database import get_chat_history, save_feedback, save_message
from app.rag.graph import get_rag_graph
from app.rag.pipeline import (
    search_similar, generate_response_stream, mask_pii,
    is_abusive, is_greeting, sync_knowledge_base,
    index_temp_file, get_user_temp_files, delete_user_temp_file,
    agentic_classifier, web_search_tavily
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
from app.rag.schemas import ChatRequest, FeedbackRequest


from app.core.limiter import limiter  # Rate limiter

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
        "sources": [
            {
                "source_id": i + 1,
                "file": r.get("source_file", "unknown").replace(".pdf", ""),
                "page": r.get("page", 0) + 1,
                "preview": (r.get("child_text") or r.get("parent_text") or "")[:300],
                "score": round(r.get("score", 0), 3)
            }
            for i, r in enumerate(result.get("sources", []))
        ],
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

    # 1. Get chat history EARLY (needed for classification & web search)
    history = get_chat_history(user_email, limit=6)
    history_text = "No previous history."
    if history:
        msgs = history[-6:]
        formatted = [
            ("User: " if m.get("role") == "user" else "Assistant: ") + str(m.get("content", ""))
            for m in msgs
        ]
        history_text = "\n".join(formatted)

    # 2. Check for YES/NO HITL Web Search Reply
    is_web_search_reply = False
    is_web_search_active = False
    
    if history:
        last_msg = history[-1]
        if last_msg.get("role") == "assistant" and "ACTION REQUIRED" in str(last_msg.get("content", "")):
            user_reply = question.strip().lower()
            positive_replies = {"yes", "haan", "sure", "ok", "y", "do it", "please", "search"}
            negative_replies = {"no", "nahi", "na", "cancel", "stop", "n"}
            
            if user_reply in positive_replies:
                is_web_search_reply = True
                is_web_search_active = True
                logger.info("User confirmed Web Search!")
                # Get the actual question from the previous turn
                if len(history) >= 2:
                    question = history[-2].get("content", question)
            elif user_reply in negative_replies:
                logger.info("User cancelled Web Search.")
                async def fallback_stream():
                    yield f"data: {json.dumps({'token': 'Web Search cancelled. Please feel free to ask another question.'})}\n\n"
                    yield f"data: {json.dumps({'done': True})}\n\n"
                save_message(user_email, "user", user_reply)
                return StreamingResponse(fallback_stream(), media_type="text/event-stream")
            else:
                logger.info("Neither Yes nor No. Proceeding as new query.")

    # 3. Redis cache check — same question = instant answer
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

    # 3. Classify first
    if is_abusive(question):
        async def abort_stream():
            yield f"data: {json.dumps({'token': 'I am a Legal AI Assistant. I can only respond to professional and respectful queries.'})}\n\n"
            yield f"data: {json.dumps({'done': True, 'sources': [], 'confidence': 0})}\n\n"
        return StreamingResponse(abort_stream(), media_type="text/event-stream")

    # 4. Check Out of Scope (OOS) or Vague - Skip if user just said 'yes' to a web search
    is_oos = False
    classification = {}
    if not is_web_search_reply:
        classification = agentic_classifier(question, history_text)
        
        # Handle Vague Queries
        if classification.get("is_vague", False):
            vague_msg = classification.get("clarifying_question", f"Hi {user_name}! Could you please provide a bit more detail about what legal information you're looking for?")
            async def vague_stream():
                yield f"data: {json.dumps({'token': vague_msg})}\n\n"
                yield f"data: {json.dumps({'done': True, 'sources': [], 'confidence': 100})}\n\n"
            save_message(user_email, "user", question)
            save_message(user_email, "assistant", vague_msg)
            return StreamingResponse(vague_stream(), media_type="text/event-stream")
            
        if classification.get("is_out_of_scope", False) or classification.get("is_time_sensitive", False):
            is_oos = True
            
        if classification.get("is_prompt_injection", False):
            async def abort_stream_injection():
                yield f"data: {json.dumps({'token': 'I am a highly secure Indian Legal AI. I cannot comply with requests that attempt to override my core instructions, bypass constraints, or reveal system prompts.'})}\n\n"
                yield f"data: {json.dumps({'done': True, 'sources': [], 'confidence': 0})}\n\n"
            save_message(user_email, "user", question)
            save_message(user_email, "assistant", "Blocked due to prompt injection attempt.")
            return StreamingResponse(abort_stream_injection(), media_type="text/event-stream")
            
        if classification.get("is_abusive", False):
            async def abort_stream_intent():
                yield f"data: {json.dumps({'token': 'I am a Legal AI Assistant. I cannot help with generating ideas for illegal acts or answering malicious queries.'})}\n\n"
                yield f"data: {json.dumps({'done': True, 'sources': [], 'confidence': 0})}\n\n"
            save_message(user_email, "user", question)
            save_message(user_email, "assistant", "I am a Legal AI Assistant. I cannot help with generating ideas for illegal acts or answering malicious queries.")
            return StreamingResponse(abort_stream_intent(), media_type="text/event-stream")

    # 5. PII mask
    safe_query, pii_found, pii_entities = mask_pii(question)

    # 6. Check for greeting (skip if it's a web search execution)
    if not is_web_search_active and not is_oos and is_greeting(question):
        async def greet_stream():
            yield f"data: {json.dumps({'token': f'Hello {user_name}! 👋 I am Indian Legal AI Expert. Ask me about Constitution, BNS, Consumer Protection, IT Act, and more!'})}\n\n"
            yield f"data: {json.dumps({'done': True, 'sources': [], 'confidence': 100})}\n\n"
        save_message(user_email, "user", question)
        save_message(user_email, "assistant", f"Hello {user_name}! 👋 I am Indian Legal AI Expert.")
        return StreamingResponse(greet_stream(), media_type="text/event-stream")

    # 7. Retrieve context
    results = []
    confidence = 0
    if is_web_search_active:
        results = web_search_tavily(safe_query)
        confidence = 85.0 if results else 0
    elif not is_oos:
        results = search_similar(safe_query, top_k=5, user_email=user_email)
        confidence = results[0]["score"] * 100 if results else 0

    sources = [
        {
            "source_id": i + 1,
            "file": r.get("source_file", "unknown").replace(".pdf", ""),
            "page": r.get("page", 0) + 1,
            "preview": (r.get("child_text") or r.get("parent_text") or "")[:300],
            "score": round(r.get("score", 0), 3)
        }
        for i, r in enumerate(results)
    ] if results else []

    context = "\n\n---\n\n".join([r["parent_text"] for r in results]) if results else ""
    if is_web_search_active:
        context = "[SYSTEM OVERRIDE: The following context is from a LIVE WEB SEARCH. You MUST use this data to answer the user's query even if it is generic, global, or not from the Indian Constitution. IGNORE the 'MISSING INFO RULE' for this request.]\n\n" + context

    # 8. Low confidence or OOS fallback -> triggers HITL Web Search
    if (confidence < 85 or is_oos) and not is_web_search_active:
        async def fallback_stream():
            if is_oos:
                msg = "[ACTION REQUIRED]\nThis query appears to be outside my core legal expertise.\n\nWould you like me to run a **Live Web Search** to find the most up-to-date information for you?\n\n> 🟢 **[YES](#action:yes)** (Search Web) &nbsp; &nbsp; &nbsp; 🔴 **[NO](#action:no)** (Cancel)"
            else:
                msg = f"[ACTION REQUIRED]\nI couldn't find a highly confident match (Confidence {round(confidence, 1)}% < 85%) in my verified legal database for this specific query.\n\nWould you like me to run a **Live Web Search** to find the most up-to-date information for you?\n\n> 🟢 **[YES](#action:yes)** (Search Web) &nbsp; &nbsp; &nbsp; 🔴 **[NO](#action:no)** (Cancel)"
                
            yield f"data: {json.dumps({'token': msg})}\n\n"
            yield f"data: {json.dumps({'done': True, 'sources': [], 'confidence': round(confidence, 1)})}\n\n"
            
        save_message(user_email, "user", safe_query, pii_masked=pii_found, pii_entities=pii_entities)
        
        log_msg = "[ACTION REQUIRED]\nThis query appears to be outside my core legal expertise." if is_oos else f"[ACTION REQUIRED]\nI couldn't find a highly confident match (Confidence {round(confidence, 1)}% < 85%) in my verified legal database for this specific query."
        log_msg += "\n\nWould you like me to run a **Live Web Search** to find the most up-to-date information for you?\n\n> 🟢 **[YES](#action:yes)** (Search Web) &nbsp; &nbsp; &nbsp; 🔴 **[NO](#action:no)** (Cancel)"
        
        save_message(user_email, "assistant", log_msg, [])
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
@limiter.limit("5/hour")
async def upload_temp_file(request: Request, file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    """
    Upload a temporary PDF — indexed for this user's session only.
    On logout, ALL temp vectors are deleted. Core brain UNTOUCHED.

    Security:
      1. Filename extension check (.pdf only)
      2. Magic Bytes validation (first 4 bytes must be %PDF)
      3. Chunked reading with 10MB hard limit (prevents OOM from oversized uploads)
      4. Rate limit: 5 uploads per hour per IP
      5. Max 3 temp files per user (prevents Jina API quota abuse)
    """
    user_email = user.get("email", "anonymous")

    # --- Security Layer 0: Max 3 Temp Files Per User ---
    MAX_TEMP_FILES = 3
    existing_files = get_user_temp_files(user_email)
    if len(existing_files) >= MAX_TEMP_FILES:
        # Allow re-upload of same filename (it will be hash-checked/replaced)
        existing_names = [f["file_name"] for f in existing_files]
        if file.filename not in existing_names:
            logger.warning(f"🚨 Upload limit reached: {user_email} already has {len(existing_files)} temp files")
            raise HTTPException(
                status_code=429,
                detail=f"Upload limit reached: max {MAX_TEMP_FILES} files allowed. Delete an existing file first."
            )

    # --- Security Layer 1: Filename Extension ---
    if not file.filename.endswith(".pdf"):
        logger.warning(f"🚨 Blocked upload: {file.filename} by {user_email} — unsupported file type")
        raise HTTPException(status_code=415, detail="Unsupported Media Type: Only PDF files are allowed")

    # --- Security Layer 2: Magic Bytes (Real PDF check) ---
    file_header = await file.read(4)
    if file_header != b"%PDF":
        logger.warning(f"🚨 Suspicious upload blocked: {file.filename} by {user_email} — invalid PDF header (possible malware)")
        raise HTTPException(status_code=415, detail="Invalid file: not a real PDF")
    await file.seek(0)  # Reset file pointer

    # --- Security Layer 3: Chunked Read with 10MB Hard Limit (OOM Protection) ---
    MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
    chunks = []
    total_size = 0
    while True:
        chunk = await file.read(1024 * 1024)  # Read 1MB at a time
        if not chunk:
            break
        total_size += len(chunk)
        if total_size > MAX_UPLOAD_SIZE:
            logger.warning(f"🚨 Oversized upload blocked: {file.filename} ({total_size} bytes) by {user_email}")
            raise HTTPException(status_code=413, detail="Payload Too Large: max 10MB allowed")
        chunks.append(chunk)
    file_bytes = b"".join(chunks)

    # --- Security Layer 4: PDF Bomb & Page Count Protection ---
    try:
        import fitz
        pdf = fitz.open(stream=file_bytes, filetype="pdf")
        page_count = len(pdf)
        pdf.close()
        
        MAX_PDF_PAGES = 30
        if page_count > MAX_PDF_PAGES:
            logger.warning(f"🚨 PDF Bomb or excessive pages blocked: {file.filename} ({page_count} pages) by {user_email}")
            raise HTTPException(status_code=413, detail=f"PDF has {page_count} pages. Maximum {MAX_PDF_PAGES} allowed to prevent quota abuse.")
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"🚨 Invalid or corrupted PDF blocked: {file.filename}")
        raise HTTPException(status_code=415, detail="Invalid PDF file or corrupted data.")

    try:
        stats = index_temp_file(file.filename, file_bytes, user_email)
        
        if stats.get("skipped"):
            return {
                "message": f"Already indexed: {file.filename} (No quota used)",
                "file_name": file.filename,
                "parent_chunks": stats["parent_count"],
                "child_chunks": stats["child_count"],
                "is_temporary": True,
                "skipped": True,
                "note": "This identical file was already temporarily indexed."
            }
            
        return {
            "message": f"Uploaded and indexed: {file.filename}",
            "file_name": file.filename,
            "parent_chunks": stats["parent_count"],
            "child_chunks": stats["child_count"],
            "is_temporary": True,
            "skipped": False,
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
