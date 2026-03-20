import asyncio
from uuid import UUID
import logging
 
from fastapi import HTTPException
from qdrant_client import AsyncQdrantClient

from sqlalchemy.ext.asyncio import AsyncSession
 
from api.modules.retrievals.retrieval_schema import (
    CollectionStatsSchema,
    RetrievalRequestSchema,
    RetrievalResponseSchema,
)
from api.modules.retrievals.retrievers.retriever_factory import RetrieverFactory
from api.modules.embedding_model_api_keys.embedding_model_key_repository import EmbeddingModelKeyRepository
from api.modules.cache.redis_service import redis_service
from api.configs.settings import settings
from shared.keys import decrypt_secret
from shared.str_normalizers import str_normalizer
from shared.vector_details import create_collection_name
import aiohttp

logger = logging.getLogger(__name__)
 
class RetrieveEmbeddingsService:
    def __init__(self, qdrant: AsyncQdrantClient, db: AsyncSession) -> None:
        self.qdrant = qdrant
        self.embedding_model_repo = EmbeddingModelKeyRepository(db)


    async def test_retrieval(self, chatbot_id: UUID, payload: RetrievalRequestSchema):
        try:
            prefix = "test_retrieval"
            model_details_in_redis = await redis_service.get_hash(
                key=str(chatbot_id), prefix=prefix
            )

            model_details = model_details_in_redis
            if not model_details:
                model_details = await self.embedding_model_repo.get_embedding_model_details(chatbot_id)
                if not model_details:
                    raise HTTPException(status_code=404, detail="Chatbot not found.")

            api_key = decrypt_secret(
                encrypted_key=model_details["api_key_encrypted"],
                encryption_key=settings.ENCRYPTION_KEY
            )
            if not api_key:
                raise HTTPException(status_code=401, detail="Invalid API key")

            vector_embeddings: RetrievalResponseSchema = await self.retrieve_embeddings(
                chatbot_id=chatbot_id,
                provider=model_details["provider"],
                api_key=api_key,
                model_name=model_details["embedding_model_name"],
                cache_prefix="test_query_embeddings",
                payload=payload
            )

            if not model_details_in_redis:
                i = 0
                is_cached = False 
                while not is_cached and i < 3:
                    is_cached = await redis_service.set_hash(
                        key=str(chatbot_id), data=model_details, prefix=prefix, ttl=3600
                    )
                    i += 1

            return vector_embeddings.model_dump(exclude_none=True)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error for testing retrieval: {e}")
            raise e


    async def retrieve_embeddings(
        self,
        chatbot_id: UUID,
        provider: str,
        api_key: str, # raw embedding model api_key
        model_name: str,
        payload: RetrievalRequestSchema,
        cache_prefix: str = "query_embeddings",
    ) -> RetrievalResponseSchema | None:
        """
        Can be use for test retrieval chatbox,
        Main retrieval service method
        """
        try:
            cached_key = str_normalizer.normalize_query_cache_key(
                prefix=f"{str(chatbot_id)}", query=payload.query
            )
            
            # check cached first
            cached_embeddings = await redis_service.get(
                key=cached_key, prefix=cache_prefix
            )
            
            if cached_embeddings:
                return RetrievalResponseSchema.model_validate_json(cached_embeddings)

            # cached not found, call retriever
            retriever = RetrieverFactory.create_retrieval(
                provider=provider,
                api_key=api_key,
                model_name=model_name,
                qdrant=self.qdrant,
            )
            
            response: RetrievalResponseSchema = await retriever.retrieve_embeddings(
                query=payload.query,
                chatbot_id=chatbot_id,
                # filters=payload.filters,
                top_k=payload.top_k,
            )
            
            if not response:
                return None
            
            # store embeddings as json in redis
            # fire and forget - don't block response on cache write
            asyncio.create_task(
                redis_service.set(
                    key=cached_key,
                    value=response.model_dump_json(),
                    prefix=cache_prefix,
                    ttl=3600
                )
            )
                    
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error for retrieval: {e}")
            raise e
    
    
    async def get_collection_stats(self, chatbot_id: UUID) -> CollectionStatsSchema | None:
        try:
            collection_name = create_collection_name(chatbot_id)
            redis_prefix = "collection_stats"

            # check redis first
            cached = await redis_service.get(key=collection_name, prefix=redis_prefix)
            if cached:
                return CollectionStatsSchema.model_validate_json(cached)

            # use qdrant client directly — no aiohttp needed
            collection_info = await self.qdrant.get_collection(collection_name)

            vectors_config = collection_info.config.params.vectors
            model_name, vector_params = next(iter(vectors_config.items()))
            model_size = vector_params.size
            model_distance = vector_params.distance.name

            points_count = collection_info.points_count
            estimated_kb = round((points_count * model_size * 4) / 1024, 2)

            stats = CollectionStatsSchema(
                model_name=model_name,
                model_size=model_size,
                model_distance=model_distance,
                total_documents=points_count,
                storage_kb=int(estimated_kb),
            )

            # cache for 3 days
            await redis_service.set(
                    key=collection_name,
                    value=stats.model_dump_json(),
                    prefix=redis_prefix,
                    ttl=259200
                )

            return stats

        except Exception as e:
            logger.warning(f"Error fetching collection stats: {e}")
            return None
            