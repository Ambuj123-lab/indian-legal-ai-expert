"""
General utility and helper functions
"""
import hashlib

def calculate_sha256(file_bytes: bytes) -> str:
    """Calculate SHA-256 hash of file contents."""
    sha256 = hashlib.sha256()
    sha256.update(file_bytes)
    return sha256.hexdigest()
