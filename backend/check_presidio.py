import spacy
import sys

try:
    nlp = spacy.load("en_core_web_sm")
    print("✅ Model 'en_core_web_sm' loaded successfully.")
    print(f"Path: {nlp.path}")
    print("Size estimate: ~12MB (Lightweight)")
except OSError:
    print("❌ Model 'en_core_web_sm' not found.")
    print("Run: python -m spacy download en_core_web_sm")
except Exception as e:
    print(f"❌ Error: {e}")
