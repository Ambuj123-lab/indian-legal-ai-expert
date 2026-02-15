import sys
import os

# Add backend to path
sys.path.append(os.getcwd())

print("Attempting to import app.main...")
try:
    from app.main import app
    print("✅ app.main imported successfully.")
except Exception as e:
    print(f"❌ Failed to import app.main: {e}")
    import traceback
    traceback.print_exc()

print("\nAttempting to import app.rag.routes...")
try:
    from app.rag.routes import router
    print("✅ app.rag.routes imported successfully.")
except Exception as e:
    print(f"❌ Failed to import app.rag.routes: {e}")
    import traceback
    traceback.print_exc()
