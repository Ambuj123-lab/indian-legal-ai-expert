"""
Quick DB Check — Verify Supabase + Qdrant sync status
Run: .\venv\Scripts\python.exe check_db.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.db.supabase_client import get_supabase
from app.rag.pipeline import get_qdrant_client, settings

def check_status():
    print("\n--- 🔍 Checking Sync Status ---")

    # Supabase
    try:
        sb = get_supabase()
        res = sb.table("document_registry").select("*").execute()
        files = res.data
        print(f"\n✅ Supabase (Document Registry): {len(files)} files")
        for f in files:
            print(f"   - {f['file_name']} ({f['chunk_count']} chunks) -> {f['status']}")
    except Exception as e:
        print(f"❌ Supabase Error: {e}")

    # Qdrant
    try:
        client = get_qdrant_client()
        coll = client.get_collection(settings.QDRANT_COLLECTION_NAME)
        print(f"\n✅ Qdrant (Vector DB): {coll.points_count} vectors")
    except Exception as e:
        print(f"❌ Qdrant Error: {e}")

    print("\n-------------------------------\n")

if __name__ == "__main__":
    check_status()
