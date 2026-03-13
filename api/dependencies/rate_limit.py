import logging
from uuid import UUID
from fastapi import Depends, HTTPException, Request
from upstash_ratelimit.asyncio import Ratelimit, SlidingWindow
from api.configs.settings import settings

from api.configs.redis import redis
from api.dependencies.auth import get_current_user

logger = logging.getLogger(__name__)


def rate_limit_by_user(
    max_request=10, window=60
):
    """Rate Limit for Authenticated User (routes with privileged access )"""
    auth_limiter = Ratelimit(
        redis=redis,
        limiter=SlidingWindow(max_requests=max_request, window=window),
        prefix="rl:auth"
    )
    
    async def dependency(request: Request, current_user_id: UUID = Depends(get_current_user)):
        try:
            response = await auth_limiter.limit(f"{request.url.path}:{str(current_user_id)}")
            if not response.allowed:
                raise HTTPException(status_code=429, detail="Too many requests.")
        except HTTPException:
            raise
        except Exception:
            logger.warning("Rate limiter unavailable (user), failing open.")
    return dependency


def rate_limit_by_api_key(
    max_request=60, window=60
):
    """Rate Limit for APIs with secret key like public chatbot"""
    api_key_limiter = Ratelimit(
        redis=redis,
        limiter=SlidingWindow(max_requests=max_request, window=window),
        prefix="rl:apikey"
    )
    
    async def dependency(request: Request):
        api_key = request.headers.get(settings.API_KEY_HEADER_NAME, "anonymous")
        try:
            response = await api_key_limiter.limit(f"{request.url.path}:{api_key}")
            if not response.allowed:
                raise HTTPException(status_code=429, detail="Too many requests.")
        except HTTPException:
            raise
        except Exception:
            logger.warning("Rate limiter unavailable (api_key), failing open.")
    return dependency


def rate_limit_by_ip(
    max_request=10, window=60
):
    """Rate Limit for API Key based limiting like chatbot product"""
    ip_limiter = Ratelimit(
        redis=redis,
        limiter=SlidingWindow(max_requests=max_request, window=window),
        prefix="rl:ip"
    )
    
    async def dependency(request: Request):
        ip = request.client.host
        try:
            response = await ip_limiter.limit(f"{request.url.path}:{ip}")
            if not response.allowed:
                raise HTTPException(status_code=429, detail="Too many requests.")
        except HTTPException:
            raise
        except Exception:
            logger.warning("Rate limiter unavailable (ip), failing open.")
    return dependency
