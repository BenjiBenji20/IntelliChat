import logging
from typing import Any
from uuid import UUID
from upstash_redis.asyncio import Redis

from api.configs.settings import settings

logger = logging.getLogger(__name__)


class RedisService:
    def __init__(self):
        self._client = Redis(
            url=settings.UPSTASH_REDIS_URL,
            token=settings.UPSTASH_REDIS_TOKEN
        )

    # -------------------------
    # GET 
    # -------------------------
    async def get(self, key: str, prefix: str = "") -> Any | None:
        """Get a value by key. Returns None on miss or error."""
        full_key = self._build_key(prefix, key)
        try:
            return await self._client.get(full_key)
        except Exception as e:
            logger.warning(f"Redis GET failed for key '{full_key}': {e}")
            return None
        
    async def get_many(self, keys: list[str], prefix: str = "") -> list[Any]:
        """Fetch multiple keys in one round trip."""
        full_keys = [self._build_key(prefix, k) for k in keys]
        try:
            pipeline = self._client.pipeline()
            for key in full_keys:
                pipeline.get(key)
            results = await pipeline.exec()
            return results
        except Exception as e:
            logger.warning(f"Redis pipeline GET failed: {e}")
            return [None] * len(keys)


    # -------------------------
    # SET 
    # -------------------------
    async def set(
        self,
        key: str,
        value: Any,
        prefix: str = "",
        ttl: int | None = None,
    ) -> bool:
        """Set key only if it doesn't exist. Returns True if set, False if already existed."""
        full_key = self._build_key(prefix,  key)
        try:
            if ttl:
                await self._client.set(full_key, value, ex=ttl, nx=True)
            else:
                await self._client.set(full_key, value, nx=True)
            return True
        except Exception as e:
            logger.warning(f"Redis SET failed for key '{full_key}': {e}")
            return False
        
    async def set_many(
        self,
        items: dict[str, Any],
        prefix: str = "",
        ttl: int | None = None,
    ) -> bool:
        """Set multiple key-value pairs in one round trip."""
        try:
            pipeline = self._client.pipeline()
            for key, value in items.items():
                full_key = self._build_key(prefix, key)
                if ttl:
                    pipeline.set(full_key, value, ex=ttl, nx=True)
                else:
                    pipeline.set(full_key, value, nx=True)
            await pipeline.exec()
            return True
        except Exception as e:
            logger.warning(f"Redis pipeline SET failed: {e}")
            return False
        
        
    async def set_hash(
        self,
        key: str,
        data: dict[str, Any],
        prefix: str = "",
        ttl: int | None = None,
    ) -> bool:
        """Store a dict as a Redis Hash."""
        full_key = self._build_key(prefix, key)
        try:
            # Serialize all values to string
            serialized = {k: str(v) for k, v in data.items()}
            await self._client.hset(full_key, values=serialized)
            if ttl:
                await self._client.expire(full_key, ttl)
            return True
        except Exception as e:
            logger.warning(f"Redis HSET failed for key '{full_key}': {e}")
            return False


    # -------------------------
    # DELETE 
    # -------------------------
    async def delete(self, key: str, prefix: str = "") -> bool:
        """Delete a key."""
        full_key = self._build_key(prefix,  key)
        try:
            await self._client.delete(full_key)
            return True
        except Exception as e:
            logger.warning(f"Redis DELETE failed for key '{full_key}': {e}")
            return False
    
    async def delete_many(self, keys: list[str], prefix: str = "") -> bool:
        """Delete multiple keys in one round trip."""
        full_keys = [self._build_key(prefix, k) for k in keys]
        try:
            pipeline = self._client.pipeline()
            for key in full_keys:
                pipeline.delete(key)
            await pipeline.exec()
            return True
        except Exception as e:
            logger.warning(f"Redis pipeline DELETE failed: {e}")
            return False
        
        
    async def invalidate_chatbot_config_data_cache(self, chatbot_id: UUID) -> None:
        """
        Call this from any service that updates:
            - LLM API keys
            - Embedding model API keys
            - Chatbot behavior / system prompt

        The next chat() or test_chat() request will re-fetch from DB and
        repopulate the cache automatically.
        """
        deleted = await self._client.delete(key=str(chatbot_id), prefix="chatbot_config_data")
        if deleted:
            logger.info(f"[CACHE INVALIDATED] chatbot_config_data for chatbot {chatbot_id}.")
        else:
            logger.warning(f"[CACHE INVALIDATE FAILED] chatbot_config_data for chatbot {chatbot_id}.")

    # -------------------------
    # SUPPORTING METHODS
    # -------------------------
    async def exists(self, key: str, prefix: str = "") -> bool:
        """Check if a key exists."""
        full_key = self._build_key(prefix, key)
        try:
            result = await self._client.exists(full_key)
            return bool(result)
        except Exception as e:
            logger.warning(f"Redis EXISTS failed for key '{full_key}': {e}")
            return False

    async def expire(self, key: str, ttl: int, prefix: str = "") -> bool:
        """Set TTL on an existing key."""
        full_key = self._build_key(prefix, key)
        try:
            await self._client.expire(full_key, ttl)
            return True
        except Exception as e:
            logger.warning(f"Redis EXPIRE failed for key '{full_key}': {e}")
            return False
        
        
    async def get_hash(
        self,
        key: str,
        prefix: str = "",
    ) -> dict[str, Any] | None:
        """Retrieve a full Hash as a dict."""
        full_key = self._build_key(prefix, key)
        try:
            result = await self._client.hgetall(full_key)
            return result if result else None
        except Exception as e:
            logger.warning(f"Redis HGETALL failed for key '{full_key}': {e}")
            return None


    async def get_hash_field(
        self,
        key: str,
        field: str,
        prefix: str = "",
    ) -> Any | None:
        """Retrieve a single field from a Hash."""
        full_key = self._build_key(prefix, key)
        try:
            return await self._client.hget(full_key, field)
        except Exception as e:
            logger.warning(f"Redis HGET failed for key '{full_key}': {e}")
            return None


    # -------------------------
    # Helpers
    # -------------------------
    @staticmethod
    def _build_key(prefix: str, key: str) -> str:
        return f"{prefix}:{key}" if prefix else key


redis_service = RedisService()
