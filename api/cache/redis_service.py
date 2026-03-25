import json
import logging
from typing import Any
from uuid import UUID
from upstash_redis.asyncio import Redis

from api.configs.settings import settings

logger = logging.getLogger(__name__)

"""
    caching prefixes:
        Frequent data query: 
            key={project_id} or key={chatbot_id}, prefix=freq_data_(data info)
        API keys:
            key={chatbot_id}. prefix=api_key_(key info)
        Secret keys:
            key={project_id} or key={chatbot_id}, prefix=secret_key
        Embeddings:
            key={chatbot_id}_{normalized_user_query}, prefix=query_embeddings
        Chat memory:
            key={chatbot_id_session_id}, prefix=chat_memory_(turn) or chat_memory_(summary)
        Tests:
            key={chatbot_id}_{}, prefix=test_(test info)
            
"""
# FREQUENT QUERY
FREQ_CACHE_PREFIX = "freq_data_"
FREQ_CACHE_TTL = 43_200 # 12 hours

# API KEY
API_KEY_CACHE_PREFIX = "api_key_"
API_KEY_CACHE_TTL    = 43_200       # 12 hours

# SECRET KEY
# SECRET_KEY_CACHE_PREFIX = "secret_key"

# TEST
TEST_CACHE_PREFIX = "test_"
TEST_CACHE_TTL = 600

# EMBEDDINGDS
EMBEDDING_CACHE_PREFIX = "query_embeddings"
EMBEDDING_CACHE_TTL = 600 # 10mins

# CHAT MEMORY
CHAT_MEMORY_CACHE_PREFIX = "chat_memory_"
SUMMARY_CHAT_MEMORY_CACHE_TTL = 43_200 # 12 hours
TURNS_MEMORY_CACHE_TTL = 604_800 # 7 days


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
        nx: bool = True,
    ) -> bool:
        """Set key only if it doesn't exist. Returns True if set, False if already existed."""
        full_key = self._build_key(prefix,  key)
        try:
            if ttl:
                await self._client.set(full_key, value, ex=ttl, nx=nx)
            else:
                await self._client.set(full_key, value, nx=nx)
            return True
        except Exception as e:
            logger.warning(f"Redis SET failed for key '{full_key}': {e}")
            return False
        
    async def set_many(
        self,
        items: dict[str, Any],
        prefix: str = "",
        ttl: int | None = None,
        nx=True
    ) -> bool:
        """Set multiple key-value pairs in one round trip."""
        try:
            pipeline = self._client.pipeline()
            for key, value in items.items():
                full_key = self._build_key(prefix, key)
                if ttl:
                    pipeline.set(full_key, value, ex=ttl, nx=nx)
                else:
                    pipeline.set(full_key, value, nx=nx)
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
        
        
    async def set_nested_dict_hash(
        self,
        key: str,
        data: dict[str, Any],
        prefix: str = "",
        ttl: int | None = None,
    ) -> bool:
        """Store a dict as a Redis Hash."""
        full_key = self._build_key(prefix, key)
        try:
            # Serialize values - use JSON for nested dicts, str for primitives
            serialized = {k: json.dumps(v, default=str) for k, v in data.items()}
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
            logger.info(f"[CACHE INVALIDATED] {prefix} for chatbot {key}.")
            return True
        except Exception as e:
            logger.warning(f"[CACHE INVALIDATE FAILED] {prefix} for chatbot {key}.")
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
    
    
    def normalize_query_cache_key(self, prefix: str, query: str) -> str:
        """
        Normalize query for consistent caching
        Removes punctuation, converts to lowercase
        """
        # Remove trailing punctuation and convert to lowercase
        return f"{prefix}_{query.lower().strip().rstrip('?!.,;:_/')}"


redis_service = RedisService()
