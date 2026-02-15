import sys
import os
from app.rag.pipeline import get_qdrant_client, settings
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Mock valid settings
from dotenv import load_dotenv
load_dotenv()

def find_bnss_section():
    client = get_qdrant_client()
    
    print("Searching for BNSS chunks containing '173'...")
    
    # Scroll through BNSS chunks
    # Filter by file name
    scroll_filter = Filter(
        must=[
            FieldCondition(key="source_file", match=MatchValue(value="Bharatiya_Nagarik_Suraksha_Sanhita_2023.pdf")),
            FieldCondition(key="chunk_type", match=MatchValue(value="child"))
        ]
    )
    
    points, _ = client.scroll(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        scroll_filter=scroll_filter,
        limit=3500, # Scan ALL chunks (BNSS has ~3211)
        with_payload=True
    )
    
    found_count = 0
    with open("bnss_chunks_debug.txt", "w", encoding="utf-8") as f:
        for point in points:
            text = point.payload.get("text", "").lower()
            # Just look for 173
            if "173" in text:
                # Check if it looks like the section header
                if "information" in text or "police" in text or "report" in text:
                    log = f"Found Candidate:\nPage: {point.payload.get('page')}\nText: {point.payload.get('text')[:300]}...\n"
                    print(log)
                    f.write(log + "\n" + "-"*20 + "\n")
                    found_count += 1
                 
    print(f"Total candidates found: {found_count}")

if __name__ == "__main__":
    find_bnss_section()
