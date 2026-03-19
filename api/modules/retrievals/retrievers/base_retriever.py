from abc import ABC, abstractmethod
from uuid import UUID
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
