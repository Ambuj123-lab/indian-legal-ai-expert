import sys
import os
import asyncio
from app.rag.pipeline import search_similar

# Mock valid settings if needed or strictly rely on env
# assuming .env is loaded by app.main or we need to load it.
from dotenv import load_dotenv
load_dotenv()

def test_retrieval():
    query = "how to file a FIR"
    print(f"Testing retrieval for query: '{query}'")
    
    # Search
    results = search_similar(query, top_k=5)
    
    print(f"\nFound {len(results)} results:\n")
    for i, res in enumerate(results):
        print(f"--- Result {i+1} ---")
        print(f"Source: {res.get('source_file')}")
        print(f"Page: {res.get('page')}")
        print(f"Score: {res.get('score')}")
        print(f"Child Text Preview: {res.get('child_text', '')[:100]}...")
        print(f"Parent Text Preview: {res.get('parent_text', '')[:100]}...")
        print("------------------\n")

if __name__ == "__main__":
    test_retrieval()
