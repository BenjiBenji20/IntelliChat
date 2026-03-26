from abc import ABC, abstractmethod
from uuid import UUID
from fastapi import HTTPException
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import QueryResponse
from api.modules.retrievals.retrieval_schema import RetrievalFilter, RetrievalResponseSchema


class BaseRetriever(ABC):
    def __init__(
        self,
        api_key: str, # raw embedding model api_key
        model_name: str,
        qdrant: AsyncQdrantClient,
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.qdrant = qdrant
        self.score_drop_tolerance = 0.10 # scores to drop against best score
        self.hard_floor_threshold = 0.49 # guardrail, bellow this, retrieved docs are irrelevant
        
        
    @abstractmethod
    async def retrieve_embeddings(
        self,
        query: str,
        chatbot_id: UUID,
        top_k: int,
        filters: list[RetrievalFilter] = None,
    ) -> RetrievalResponseSchema | None:
        """Requires child class to define retrieve() method and return the same pydantic schema."""
        pass
    
    
    @abstractmethod
    def determine_score_threshold(self, knowledge_list: QueryResponse) -> float:
        """Filter retrieved results list based on relevant scores"""
        if not knowledge_list.points:
            return None

        scores = [hit.score for hit in knowledge_list.points]
        best_score = max(scores)
        
        # Early exit — if best score is lower to the hard floor threshold, no more relevant docs
        if best_score <= self.hard_floor_threshold:
            return None

        # keep docs within tolerance of the best score
        score_threshold = best_score - self.score_drop_tolerance
        
        return score_threshold if score_threshold > self.hard_floor_threshold else None
    
    
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
