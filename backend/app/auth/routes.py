"""
Auth Routes — Google OAuth Login + JWT Token Issuance
"""

import logging
import json
from datetime import datetime

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import RedirectResponse
from authlib.integrations.starlette_client import OAuthError

from app.config import get_settings
from app.auth.jwt import create_access_token
from app.auth.oauth import oauth

router = APIRouter(prefix="/auth", tags=["Authentication"])
settings = get_settings()
logger = logging.getLogger(__name__)


@router.get("/login")
async def login(request: Request):
    """Redirect to Google OAuth consent screen"""
    redirect_uri = settings.GOOGLE_REDIRECT_URI
    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get("/callback")
async def auth_callback(request: Request):
    """Handle Google OAuth callback — issue JWT and redirect to frontend"""
    try:
        # ProxyHeadersMiddleware ensures request.url is HTTPS, so Authlib infers correctly.
        token = await oauth.google.authorize_access_token(request)
        user_info = token.get('userinfo')

        if not user_info:
            raise HTTPException(status_code=400, detail="Could not get user info from Google")

        # Create JWT token with user data
        access_token = create_access_token(data={
            "email": user_info.get("email"),
            "name": user_info.get("name"),
            "picture": user_info.get("picture")
        })

        logger.info(json.dumps({
            "event": "user_login",
            "email": user_info.get("email"),
            "timestamp": datetime.now().isoformat()
        }))

        # Redirect to frontend with JWT token
        # NOTE: Uses /auth-callback (NOT /auth/callback) to avoid collision with this backend route
        return RedirectResponse(
            url=f"{settings.FRONTEND_URL}/auth-callback?token={access_token}"
        )

    except OAuthError as error:
        logger.error(f"OAuth Error: {error}")
        return RedirectResponse(
            url=f"{settings.FRONTEND_URL}/login?error={error}"
        )
    except Exception as e:
        logger.error(f"Auth Callback Error: {e}")
        return RedirectResponse(
            url=f"{settings.FRONTEND_URL}/login?error={e}"
        )


@router.get("/me")
async def get_me(request: Request):
    """Test endpoint — verify JWT works"""
    from app.auth.jwt import get_current_user
    from fastapi.security import HTTPAuthorizationCredentials
    
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="No token provided")
    
    token = auth_header.split(" ")[1]
    from app.auth.jwt import verify_token
    payload = verify_token(token)
    return {"user": payload}


@router.post("/logout")
async def logout(request: Request):
    """
    Logout — cleanup user's temporary vectors from Qdrant.
    Core brain (is_temporary=False) is NEVER touched.
    """
    deleted_count = 0

    # Try to extract user email from JWT for cleanup
    try:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            from app.auth.jwt import verify_token
            payload = verify_token(token)
            user_email = payload.get("email", "")

            if user_email:
                from app.rag.pipeline import cleanup_user_temp_vectors
                deleted_count = cleanup_user_temp_vectors(user_email)

                logger.info(json.dumps({
                    "event": "user_logout",
                    "email": user_email,
                    "temp_vectors_deleted": deleted_count,
                    "timestamp": datetime.now().isoformat()
                }))
    except Exception as e:
        logger.warning(f"Logout cleanup error (non-critical): {e}")

    return {
        "message": "Logged out successfully",
        "temp_vectors_cleaned": deleted_count
    }
