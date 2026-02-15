"""
Supabase Client — Document Registry + Storage Operations
Handles: SHA-256 tracking, incremental sync, orphan cleanup
"""

import hashlib
import logging
import json
from datetime import datetime
from typing import Optional

from supabase import create_client, Client
from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Global Supabase client
_supabase: Optional[Client] = None

BUCKET_NAME = "legal-docs"
TABLE_NAME = "document_registry"


def get_supabase() -> Optional[Client]:
    """Get or create Supabase client"""
    global _supabase
    if _supabase is None:
        try:
            if settings.SUPABASE_URL and settings.SUPABASE_KEY:
                _supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)
                logger.info("Supabase client initialized")
            else:
                logger.warning("Supabase URL/KEY not set — document management disabled")
        except Exception as e:
            logger.error(f"Supabase init failed: {e}")
    return _supabase


def calculate_sha256(file_bytes: bytes) -> str:
    """Calculate SHA-256 hash of file content"""
    return hashlib.sha256(file_bytes).hexdigest()


# --- Storage Operations ---

def list_storage_files() -> list:
    """List all files in Supabase Storage bucket"""
    client = get_supabase()
    if not client:
        return []
    try:
        files = client.storage.from_(BUCKET_NAME).list()
        # Filter out folders, keep only files
        return [f for f in files if f.get("name") and not f.get("name", "").endswith("/")]
    except Exception as e:
        logger.error(f"Storage list error: {e}")
        return []


def download_file(file_name: str) -> Optional[bytes]:
    """Download file from Supabase Storage"""
    client = get_supabase()
    if not client:
        return None
    try:
        response = client.storage.from_(BUCKET_NAME).download(file_name)
        return response
    except Exception as e:
        logger.error(f"Storage download error for {file_name}: {e}")
        return None


def delete_storage_file(file_name: str) -> bool:
    """Delete file from Supabase Storage"""
    client = get_supabase()
    if not client:
        return False
    try:
        client.storage.from_(BUCKET_NAME).remove([file_name])
        logger.info(f"Deleted from storage: {file_name}")
        return True
    except Exception as e:
        logger.error(f"Storage delete error for {file_name}: {e}")
        return False


# --- Document Registry Operations ---

def get_all_registry_entries() -> list:
    """Get all entries from document_registry table"""
    client = get_supabase()
    if not client:
        logger.error("get_all_registry_entries: Supabase client is None")
        return []
    try:
        response = client.table(TABLE_NAME).select("*").execute()
        data = response.data or []
        logger.info(f"Registry fetch: {len(data)} documents found.")
        return data
    except Exception as e:
        logger.error(f"Registry read error: {e}")
        import traceback
        traceback.print_exc()
        return []


def get_registry_entry(file_name: str) -> Optional[dict]:
    """Get single entry by filename"""
    client = get_supabase()
    if not client:
        return None
    try:
        response = client.table(TABLE_NAME).select("*").eq("file_name", file_name).execute()
        if response.data:
            return response.data[0]
        return None
    except Exception as e:
        logger.error(f"Registry lookup error for {file_name}: {e}")
        return None


def upsert_registry_entry(
    file_name: str,
    file_hash: str,
    file_size: int,
    chunk_count: int = 0,
    parent_chunk_count: int = 0,
    child_chunk_count: int = 0,
    status: str = "active"
):
    """Insert or update a document registry entry"""
    client = get_supabase()
    if not client:
        return
    try:
        data = {
            "file_name": file_name,
            "file_hash": file_hash,
            "file_size": file_size,
            "chunk_count": chunk_count,
            "parent_chunk_count": parent_chunk_count,
            "child_chunk_count": child_chunk_count,
            "status": status,
            "storage_path": f"{BUCKET_NAME}/{file_name}",
            "updated_at": datetime.now().isoformat()
        }
        # Upsert: insert if new, update if exists
        client.table(TABLE_NAME).upsert(data, on_conflict="file_name").execute()
        logger.info(f"Registry upserted: {file_name} (status={status}, chunks={chunk_count})")
    except Exception as e:
        logger.error(f"Registry upsert error for {file_name}: {e}")


def mark_registry_deleted(file_name: str):
    """Mark a document as deleted in registry"""
    client = get_supabase()
    if not client:
        return
    try:
        client.table(TABLE_NAME).update({
            "status": "deleted",
            "updated_at": datetime.now().isoformat()
        }).eq("file_name", file_name).execute()
        logger.info(f"Registry marked deleted: {file_name}")
    except Exception as e:
        logger.error(f"Registry delete-mark error for {file_name}: {e}")


def delete_registry_entry(file_name: str):
    """Permanently remove entry from registry"""
    client = get_supabase()
    if not client:
        return
    try:
        client.table(TABLE_NAME).delete().eq("file_name", file_name).execute()
        logger.info(f"Registry entry removed: {file_name}")
    except Exception as e:
        logger.error(f"Registry delete error for {file_name}: {e}")


def get_registry_stats() -> dict:
    """Get summary stats from registry"""
    entries = get_all_registry_entries()
    active = [e for e in entries if e.get("status") == "active"]
    return {
        "total_documents": len(active),
        "total_chunks": sum(e.get("chunk_count", 0) for e in active),
        "total_parent_chunks": sum(e.get("parent_chunk_count", 0) for e in active),
        "total_child_chunks": sum(e.get("child_chunk_count", 0) for e in active),
        "documents": [
            {
                "file_name": e.get("file_name"),
                "status": e.get("status"),
                "chunk_count": e.get("chunk_count", 0),
                "file_size": e.get("file_size", 0),
                "indexed_at": e.get("indexed_at"),
                "updated_at": e.get("updated_at")
            }
            for e in entries
        ]
    }
