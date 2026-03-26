from uuid import UUID
import logging

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from api.modules.retrievals.retrievers.base_retriever import *
from api.modules.retrievals.retrieval_schema import ChunkResultSchema, RetrievalResponseSchema
from google.api_core.exceptions import Unauthenticated, NotFound, ResourceExhausted, GoogleAPIError
from qdrant_client.http.models import QueryResponse, Filter, FieldCondition, MatchValue
from shared.vector_details import create_collection_name

logger = logging.getLogger(__name__)

class GeminiRetriever(BaseRetriever):
    """
    GeminiRetriever only supports embedding models from Gemini model or
    Google AI Studio provider
    Models:
        gemini-embedding-001,
        text-embedding-005,
        text-multilingual-embedding-002,
        gemini-embedding-2-preview,    
    """
    
    async def retrieve_embeddings(
        self,
        query: str,
        chatbot_id: UUID,
        top_k: int,
        filters: list[RetrievalFilter] = None
    ) -> RetrievalResponseSchema | None:
        try:
            embedder = GoogleGenerativeAIEmbeddings(
                api_key=self.api_key,
                model=self.model_name,
                task_type="RETRIEVAL_QUERY"
            )

            query_vector: list[float] = await embedder.aembed_query(query)

            if not query_vector:
                logger.info("No embeddings found.")
                return None

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

        except Exception as e:
            logger.error(
                f"Gemini Retriever failed for chatbot_id={chatbot_id}. Error: {e}"
            )
            raise
    
        
    async def test_retrieve_embeddings(self) -> bool:
        try:
            embedder = GoogleGenerativeAIEmbeddings(
                model=self.model_name,
                google_api_key=self.api_key,
            )
            await embedder.aembed_query("hi")
            return True

        except Unauthenticated:
            raise EmbedderAuthError()
        except NotFound:
            raise EmbedderModelNotFoundError()
        except ResourceExhausted:
            raise EmbedderRateLimitError()
        except GoogleAPIError as e:
            raise EmbedderConnectionError(str(e))
