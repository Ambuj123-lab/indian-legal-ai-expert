"""
MongoDB Database — Chat History + Feedback
Reused from Citizen Safety with GDPR TTL (30 days auto-delete)
"""

import json
import logging
from datetime import datetime
from typing import List, Optional

import pymongo
from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Global instances
_client = None
_db = None


def init_mongodb():
    """Initialize MongoDB connection with GDPR TTL index"""
    global _client, _db
    try:
        if settings.MONGO_URI:
            _client = pymongo.MongoClient(settings.MONGO_URI)
            _db = _client[settings.MONGO_DB_NAME]

            # GDPR Compliance: Auto-delete after 30 days of inactivity
            _db["chat_history"].create_index("last_activity", expireAfterSeconds=2592000)

            logger.info(json.dumps({
                "event": "mongodb_connected",
                "database": settings.MONGO_DB_NAME,
                "timestamp": datetime.now().isoformat()
            }))
        else:
            logger.warning("MONGO_URI not set — chat history disabled")
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}")


def get_db():
    """Get database instance"""
    return _db


def get_chat_collection():
    """Get chat history collection"""
    if _db is not None:
        return _db["chat_history"]
    return None


def get_feedback_collection():
    """Get feedback collection"""
    if _db is not None:
        return _db["feedback"]
    return None


def save_message(
    user_email: str,
    role: str,
    content: str,
    sources: Optional[List[dict]] = None,
    pii_masked: bool = False,
    pii_entities: Optional[List[dict]] = None
):
    """Save chat message to MongoDB"""
    collection = get_chat_collection()
    if collection is None:
        logger.warning("Chat collection is None — MongoDB not connected")
        return

    try:
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        if sources:
            message["sources"] = sources
        if pii_masked:
            message["pii_masked"] = True
            message["pii_entities"] = pii_entities or []

        collection.update_one(
            {"user_email": user_email},
            {
                "$push": {"messages": message},
                "$set": {"last_activity": datetime.now()}
            },
            upsert=True
        )
    except Exception as e:
        logger.error(f"Save message error: {e}")


def get_chat_history(user_email: str, limit: int = 6) -> List[dict]:
    """Get recent chat history for a user"""
    collection = get_chat_collection()
    if collection is None:
        return []

    try:
        doc = collection.find_one({"user_email": user_email})
        if doc and "messages" in doc:
            return doc["messages"][-limit:]
        return []
    except Exception as e:
        logger.error(f"Get history error: {e}")
        return []


def save_feedback(user_email: str, question: str, response: str, rating: str):
    """Save user feedback (👍/👎)"""
    collection = get_feedback_collection()
    if collection is None:
        return

    try:
        collection.insert_one({
            "user_email": user_email,
            "question": question,
            "response": response,
            "rating": rating,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Save feedback error: {e}")
