import sys
import os
import asyncio
from app.rag.pipeline import search_similar

# Mock valid settings if needed or strictly rely on env
from dotenv import load_dotenv
load_dotenv()

def test_retrieval():
    query = "how to file a FIR"
    with open("retrieval_results_utf8.txt", "w", encoding="utf-8") as f:
        f.write(f"Testing retrieval for query: '{query}'\n")
        
        # Search
        results = search_similar(query, top_k=5)
        
        f.write(f"\nFound {len(results)} results:\n")
        for i, res in enumerate(results):
            f.write(f"--- Result {i+1} ---\n")
            f.write(f"Source: {res.get('source_file')}\n")
            f.write(f"Page: {res.get('page')}\n")
            f.write(f"Score: {res.get('score')}\n")
            f.write(f"Child Text Preview: {res.get('child_text', '')[:100]}...\n")
            f.write(f"Parent Text Preview: {res.get('parent_text', '')[:100]}...\n")
            f.write("------------------\n")

if __name__ == "__main__":
    test_retrieval()
