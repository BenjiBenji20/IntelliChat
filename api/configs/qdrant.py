from qdrant_client import AsyncQdrantClient
from api.configs.settings import settings

_qdrant_client: AsyncQdrantClient = None

async def init_qdrant_client():
    global _qdrant_client
    _qdrant_client = AsyncQdrantClient(
        url=settings.QDRANT_CLUSTER_ENDPOINT,
        api_key=settings.QDRANT_API_KEY,
    )

async def close_qdrant_client():
    global _qdrant_client
    if _qdrant_client:
        await _qdrant_client.close()

def get_qdrant_client() -> AsyncQdrantClient:
    return _qdrant_client
