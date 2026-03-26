from uuid import UUID
import logging

from langchain_openai import OpenAIEmbeddings

from api.modules.retrievals.retrievers.base_retriever import *
from api.modules.retrievals.retrieval_schema import RetrievalResponseSchema
from openai import (
    AuthenticationError,
    NotFoundError,
    RateLimitError,
    APIError
)

logger = logging.getLogger(__name__)

class OpenAIRetriever(BaseRetriever):
    """
    OpenAIRetriever only supports embedding models from OpenAI provider
    Models:
        "text-embedding-ada-002"
        "text-embedding-3-small"
        "text-embedding-3-large"   
    """
    
    async def retrieve_embeddings(
        self,
        query: str,
        chatbot_id: UUID,
        top_k: int,
        filters: list[RetrievalFilter] = None
    ) -> RetrievalResponseSchema | None:
        try:
            embedder = OpenAIEmbeddings(
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
                f"OpenAIRetriever failed for chatbot_id={chatbot_id}. Error: {e}"
            )
            raise
    
        
    async def test_retrieve_embeddings(self) -> bool:
        try:
            embedder = OpenAIEmbeddings(
                model=self.model_name,
                api_key=self.api_key,
                task_type="RETRIEVAL_QUERY"
            )
            await embedder.aembed_query("hi")
            return True

        except AuthenticationError:
            raise EmbedderAuthError()
        except NotFoundError:
            raise EmbedderModelNotFoundError()
        except RateLimitError:
            raise EmbedderRateLimitError()
        except APIError as e:
            raise EmbedderConnectionError(str(e))
