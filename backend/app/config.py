"""
Application Configuration — Pydantic Settings
Loads from .env file, centralizes all config.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # --- App ---
    APP_NAME: str = "Indian Legal AI Expert"
    DEBUG: bool = False

    # --- Google OAuth ---
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    GOOGLE_REDIRECT_URI: str = "http://localhost:8000/auth/callback"
    FRONTEND_URL: str = "http://localhost:5173"

    # --- JWT (7-day expiry) ---
    SECRET_KEY: str = "change-this-to-a-random-secret-key"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

    # --- OpenRouter LLM ---
    OPENROUTER_API_KEY: str = ""

    # --- Jina AI Embeddings ---
    JINA_API_KEY: str = ""

    # --- Qdrant Vector DB ---
    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION_NAME: str = "legal-knowledge"

    # --- Supabase (Document Registry + Storage) ---
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""  # anon public key
    SUPABASE_SERVICE_ROLE_KEY: str = ""  # service_role key (for storage admin ops)

    # --- MongoDB Atlas (Chat History + Feedback) ---
    MONGO_URI: str = ""
    MONGO_DB_NAME: str = "legal_ai_expert"

    # --- Upstash Redis (Cache + Analytics) ---
    UPSTASH_REDIS_REST_URL: str = ""
    UPSTASH_REDIS_REST_TOKEN: str = ""

    # --- Langfuse (Observability) ---
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_HOST: str = "https://us.cloud.langfuse.com"

    # --- Rate Limiting ---
    RATE_LIMIT_PER_MINUTE: int = 10

    # --- Admin Access ---
    ADMIN_EMAIL: str = ""  # Only this email can sync/delete documents

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()
