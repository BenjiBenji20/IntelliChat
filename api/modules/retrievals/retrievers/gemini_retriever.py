from uuid import UUID
import logging

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from api.modules.retrievals.retrievers.base_retriever import BaseRetriever
from api.modules.retrievals.retrieval_schema import ChunkResultSchema, RetrievalResponseSchema
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
        # filters: list[RetrievalFilter]
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

            # build Qdrant filter conditions
            # must_conditions = [
            #     FieldCondition(key=k, match=MatchValue(value=v))
            #     for f in filters
            #     for k, v in f.model_dump(exclude_none=True).items()
            #     if str(v).strip()
            # ]
            # query_filter = Filter(must=must_conditions) if must_conditions else None

            search = await self.qdrant.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=top_k,
                using=self.model_name,
                # query_filter=query_filter, # thinking of better filter strategy in future
                with_payload=True,
                with_vectors=False
            )

            chunks: list[ChunkResultSchema] = [
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
                for hit in search.points
            ]

            return RetrievalResponseSchema(
                query=query,
                top_k=top_k,
                total_results=len(chunks),
                results=chunks
            )

        except Exception as e:
            logger.error(
                f"Gemini Retriever failed for chatbot_id={chatbot_id}. Error: {e}"
            )
            raise
