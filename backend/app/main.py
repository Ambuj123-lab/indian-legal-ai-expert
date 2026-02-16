"""
FastAPI Application — Indian Legal AI Expert
Entry point with middleware, CORS, rate limiting, and startup events.
"""

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime

import os
from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.config import get_settings
from app.limiter import limiter  # Import shared limiter instance

settings = get_settings()

# --- Structured Logging ---
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# --- Lifespan (Startup / Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info(json.dumps({
        "event": "startup",
        "app": settings.APP_NAME,
        "timestamp": datetime.now().isoformat()
    }))

    # Initialize MongoDB
    from app.db.database import init_mongodb
    init_mongodb()

    # Initialize Supabase
    from app.db.supabase_client import get_supabase
    get_supabase()

    # Initialize Qdrant collection
    try:
        from app.rag.pipeline import ensure_collection
        ensure_collection()
    except Exception as e:
        logger.warning(f"Qdrant init skipped (will retry on first request): {e}")

    yield

    logger.info(json.dumps({
        "event": "shutdown",
        "timestamp": datetime.now().isoformat()
    }))


# --- App Instance ---
app = FastAPI(
    title=settings.APP_NAME,
    description="Production-grade RAG chatbot for Indian Legal queries — powered by LangGraph, Qdrant, and Jina AI",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# --- Rate Limiting ---
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.FRONTEND_URL,
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Session Middleware (for OAuth state) ---
# same_site="none" + https_only=True needed for OAuth cross-site redirect from Google
app.add_middleware(SessionMiddleware, secret_key=settings.SECRET_KEY, https_only=True, same_site="none")

# --- Proxy Headers Middleware (for Koyeb SSL termination) ---
# MUST be added LAST so it runs FIRST (Outer-most), fixing the scheme before Session sees it.
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")

# --- Include Routers ---
from app.auth.routes import router as auth_router
from app.rag.routes import router as rag_router

app.include_router(auth_router)
app.include_router(rag_router)


# --- Health Check ---
@app.get("/health")
async def health_check():
    """Health check for deployment monitoring"""
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat()
    }


# --- Root ---
@app.get("/")
async def root():
    # If running in Docker/Production, serve Frontend
    frontend_dist = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist")
    index_path = os.path.join(frontend_dist, "index.html")
    
    if os.path.exists(index_path):
        return Response(
            content=open(index_path, "rb").read(),
            media_type="text/html"
        )
    
    # Fallback for local backend-only dev
    return {
        "message": f"{settings.APP_NAME} API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/api/debug-config")
async def debug_config():
    return {
        "FRONTEND_URL": settings.FRONTEND_URL,
        "GOOGLE_REDIRECT_URI": settings.GOOGLE_REDIRECT_URI,
        "ENV_VARS": {k: v for k, v in os.environ.items() if k in ["FRONTEND_URL", "GOOGLE_REDIRECT_URI"]}
    }

# --- Serve Frontend (Production / Docker) ---
# This block runs only when deployed via Docker where frontend is built to /app/frontend/dist
frontend_dist = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist")
if os.path.exists(frontend_dist):
    # Mount static assets (JS, CSS, Images)
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="assets")
    
    # Mount branding/images if present in public folder copy
    if os.path.exists(os.path.join(frontend_dist, "branding")):
        app.mount("/branding", StaticFiles(directory=os.path.join(frontend_dist, "branding")), name="branding")

    # Serve index.html for all other routes (SPA Catch-all)
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        if full_path.startswith("api") or full_path.startswith("docs") or full_path.startswith("openapi.json"):
            return Response(status_code=404)
        return Response(
            content=open(os.path.join(frontend_dist, "index.html"), "rb").read(),
            media_type="text/html"
        )
