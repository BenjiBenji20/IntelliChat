from uuid import UUID
import logging

from langchain_cohere import CohereEmbeddings

from api.modules.retrievals.retrievers.base_retriever import *
from api.modules.retrievals.retrieval_schema import RetrievalResponseSchema
from cohere import (
    UnauthorizedError,
    NotFoundError,
    TooManyRequestsError,
    InternalServerError
)

logger = logging.getLogger(__name__)

class CohereRetriever(BaseRetriever):
    """
    CohereRetriever only supports embedding models from Cohere provider
    Models:
        "embed-v4.0",
        "embed-english-v3.0",
        "embed-english-light-v3.0",
        "embed-multilingual-v3.0",   
    """
    
    async def retrieve_embeddings(
        self,
        query: str,
        chatbot_id: UUID,
        top_k: int,
        filters: list[RetrievalFilter] = None
    ) -> RetrievalResponseSchema | None:
        try:
            embedder = CohereEmbeddings(
                api_key=self.api_key,
                model=self.model_name,
                task_type="RETRIEVAL_QUERY"
            )

            query_vector: list[float] = await embedder.aembed_query(query)

            if not query_vector:
                logger.info("No embeddings found.")
                return None

            return await self.process_semantic_search(
                query=query, chatbot_id=chatbot_id,
                top_k=top_k, query_vector=query_vector,
                filters=filters
            )

        except Exception as e:
            logger.error(
                f"CohereRetriever failed for chatbot_id={chatbot_id}. Error: {e}"
            )
            raise
    
        
    async def test_retrieve_embeddings(self) -> bool:
        try:
            embedder = CohereEmbeddings(
                model=self.model_name,
                api_key=self.api_key,
                task_type="RETRIEVAL_QUERY"
            )
            await embedder.aembed_query("hi")
            return True

        except UnauthorizedError:
            raise EmbedderAuthError()
        except NotFoundError:
            raise EmbedderModelNotFoundError()
        except TooManyRequestsError:
            raise InternalServerError()
        except InternalServerError as e:
            raise EmbedderConnectionError(str(e))
