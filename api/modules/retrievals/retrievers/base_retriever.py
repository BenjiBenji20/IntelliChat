from abc import ABC, abstractmethod
from uuid import UUID
from fastapi import HTTPException
from qdrant_client import AsyncQdrantClient
from api.modules.retrievals.retrieval_schema import ChunkResultSchema, RetrievalFilter, RetrievalResponseSchema
from qdrant_client.http.models import QueryResponse, Filter, FieldCondition, MatchValue
from shared.vector_details import create_collection_name

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
    
    
    async def process_semantic_search(
        self, 
        query: str,
        chatbot_id: UUID,
        top_k: int,
        query_vector: list[float],
        filters: list[RetrievalFilter] = None
    ) -> RetrievalResponseSchema:
        """Abstract the retrieval process. Errors catch by caller method"""
        collection_name = create_collection_name(chatbot_id)

        query_filter = None
        if filters:
            should_conditions = []
            # If multiple filters are provided, they should act as an OR condition
            for f in filters:
                must_conditions = []
                for k, v in f.model_dump(exclude_none=True).items():
                    if str(v).strip():
                        must_conditions.append(
                            FieldCondition(key=k, match=MatchValue(value=v))
                        )
                
                if must_conditions:
                    should_conditions.append(Filter(must=must_conditions))
            
            if should_conditions:
                query_filter = Filter(should=should_conditions)

        search = await self.qdrant.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            using=self.model_name,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False
        )
        
        score_threshold = self.determine_score_threshold(search)
        
        # if no valid threshold (all docs irrelevant)
        if score_threshold is None:
            return RetrievalResponseSchema(
                query=query,
                top_k=top_k,
                total_results=0,
                results=[]
            )
        
        # store relevant documents by score
        relevant_docs: list[ChunkResultSchema] = [
            ChunkResultSchema(
                chunk_id=str(hit.id),
                document_id=hit.payload.get("document_id", ""),
                chunk_index=hit.payload.get("chunk_index", 0),
                score=hit.score,
                page_content=hit.payload.get("page_content", ""),
                file_name=hit.payload.get("file_name", ""),
                file_type=hit.payload.get("file_type", ""),
                content_type=hit.payload.get("content_type", "knowledge"),
                document_type=hit.payload.get("document_type", ""),
                ingestion_time=hit.payload.get("ingestion_time", ""),
                page_number=hit.payload.get("page_number"),
                section=hit.payload.get("section"),
                heading_level=hit.payload.get("heading_level"),
                json_path=hit.payload.get("json_path"),
                record_id=hit.payload.get("record_id"),
            )
            for hit in search.points if hit.score >= score_threshold # score should be >= 0.50
        ]

        return RetrievalResponseSchema(
            query=query,
            top_k=top_k,
            total_results=len(relevant_docs),
            results=relevant_docs
        )
    
    
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
