import logging
import sys
from app.core.config import get_settings

settings = get_settings()

def setup_logging():
    """Configure structured logging for the application"""
    logging.basicConfig(
        level=logging.DEBUG if settings.DEBUG else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout     # Ensure logs go to stdout for Render/Docker to capture
    )
    
    # Silence third-party noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    
    logger = logging.getLogger("app")
    logger.info("Structured logging initialized")
    return logger
