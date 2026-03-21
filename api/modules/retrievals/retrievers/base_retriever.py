from abc import ABC, abstractmethod
from uuid import UUID
from fastapi import HTTPException
from qdrant_client import AsyncQdrantClient

from api.modules.retrievals.retrieval_schema import RetrievalResponseSchema


class BaseRetriever(ABC):
    def __init__(
        self,
        api_key: str, # raw embedding model api_key
        model_name: str,
        qdrant: AsyncQdrantClient, # AsyncQDrantClient
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.qdrant = qdrant
        
    @abstractmethod
    async def retrieve_embeddings(
        self,
        query: str,
        chatbot_id: UUID,
        top_k: int,
        # filters: list[RetrievalFilter] = None,
    ) -> RetrievalResponseSchema | None:
        """Requires child class to define retrieve() method and return the same pydantic schema."""
        pass
    
    
    @abstractmethod
    async def test_retrieve_embeddings(self):
        """Testing Embedding Model key configuration on registration"""
        pass
    
    
    @staticmethod
    def raise_http_from_retrieval_error(e: Exception):
        if isinstance(e, EmbedderAuthError):
            raise HTTPException(status_code=401, detail="Invalid Embedder API key.")
        if isinstance(e, EmbedderModelNotFoundError):
            raise HTTPException(status_code=404, detail="Model not found on this provider.")
        if isinstance(e, EmbedderRateLimitError):
            raise HTTPException(status_code=429, detail="Embedder API key rate limit exceeded.")
        if isinstance(e, EmbedderConnectionError):
            raise HTTPException(status_code=502, detail="Could not reach the Embedder provider.")
        raise HTTPException(status_code=500, detail="Unexpected Embedder error.")


# provider agnostic exception hanlers
class EmbedderAuthError(Exception): pass
class EmbedderModelNotFoundError(Exception): pass
class EmbedderRateLimitError(Exception): pass
class EmbedderConnectionError(Exception): pass
