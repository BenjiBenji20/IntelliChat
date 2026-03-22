from cryptography.fernet import Fernet
from fastapi import HTTPException, status
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncSession

from api.models.embedding_model_key import EmbeddingModelKey
from api.modules.cache.redis_service import redis_service, API_KEY_CACHE_PREFIX, FREQ_CACHE_PREFIX
from api.modules.chatbot.chatbot_repository import ChatbotRepository
from api.modules.llm_api_keys.llm_key_repository import LlmKeyRepository
from api.modules.embedding_model_api_keys.embedding_model_key_repository import EmbeddingModelKeyRepository
from api.modules.documents.chunking_config_repository import ChunkingConfigurationRepository
from api.modules.embedding_model_api_keys.embedding_model_api_keys_schema import *
from api.configs.settings import settings
from api.modules.retrievals.retrievers.base_retriever import *
from api.modules.retrievals.retrievers.retriever_factory import RetrieverFactory
from shared.ai_models_details import embedder_provider_mapper, embedder_provider_validator

from logging import getLogger

logger = getLogger(__name__)

class EmbeddingModelAPIKeyService:
    def __init__(self, db: AsyncSession, qdrant: AsyncQdrantClient):
        """
        This service will interact with 3 models: Chatbot, LLMKey and EmbeddingModelKey
        Chatbot model creation has steps:
        1. Chatbot's identity: User creates a chatbot name, prompt, etc...
        2. Chatbot's knowledge: User uploads API keys [LLM, Embedding Model]
        """
        self.db = db
        self.qdrant = qdrant
        self.chatbot_repo = ChatbotRepository(db)
        self.llm_key_repo = LlmKeyRepository(db)
        self.embedding_model_key_repo = EmbeddingModelKeyRepository(db)
        self.chunk_config_repo = ChunkingConfigurationRepository(db)
        self.cached_prefix = f"{API_KEY_CACHE_PREFIX}(chatbot_config_data)"
    
    async def upload_embedding_model_key(
        self, payload: CreateRequestEmbbedingModelSchema
    ) -> ResponseEmbbedingModelSchema:
        """
        User inputs:
            -  Embedding model API Key, model name, temperatur, provider
        Table: embedding_model_keys
        Flow:
            Test API token
            Encypt raw api key and store in payload
            Store in DB
        """
        try:
            payload_dict = payload.model_dump()
            
            raw_embedding_model_key = payload_dict['api_key']
            embedding_model_name = payload_dict["embedding_model_name"]
            provider = payload_dict["provider"]
            # get and pop project_id 
            project_id = payload_dict["project_id"] # use for invalidate caching
            payload_dict.pop("project_id")
            
            if not embedder_provider_validator(provider):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Embedder model provider not supported."
                )
            
            if not embedder_provider_mapper(embedding_model_name, provider):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Embedder model name and provider combination not supported."
                )
            
            # test the model if it has token
            await self.test_embedding_model_api_key(
                api_key=raw_embedding_model_key,
                provider=provider,
                model_name=embedding_model_name
            )
            
            if settings.ENCRYPTION_KEY is None:
                raise ValueError("Encryption key not set.")
            
            payload_dict.update({
                'api_key_encrypted': self.encrypt_api_key(
                    encryption_key=settings.ENCRYPTION_KEY,
                    api_key_string=raw_embedding_model_key
                ),
                "embedding_model_name": embedding_model_name,
                "provider": provider
            })
            
            payload_dict.pop("api_key") # remove the raw api_key
            
            embedding_model_key: EmbeddingModelKey = await self.embedding_model_key_repo.create(**payload_dict)
            
            if not embedding_model_key:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to register API key. Please try again."
                )

            # invalidate chatbot current state cache
            await redis_service.invalidate_chatbot_config_data_cache(
                key=str(project_id), prefix=f"{FREQ_CACHE_PREFIX}(chatbot_current_state)"
            )
        
            return ResponseEmbbedingModelSchema(
                id=embedding_model_key.id,
                project_id=project_id,
                user_id=embedding_model_key.user_id,
                chatbot_id=embedding_model_key.chatbot_id,
                api_key=embedding_model_key.api_key_encrypted,
                embedding_model_name=embedding_model_key.embedding_model_name,
                provider=embedding_model_key.provider,
                created_at=embedding_model_key.created_at,
                updated_at=embedding_model_key.updated_at
            )
        
        except HTTPException:
            raise
        except Exception as e:
            await self.db.rollback()
            raise e
    
    
    async def update_embedding_model_api_key(
        self, payload: UpdateRequestEmbeddingModelSchema
    ) -> tuple[ResponseEmbbedingModelSchema, list[dict] | None]:
        """
        Update: embedding_model_keys
        Not to update: id, user_id, created_at and original chatbot_id

        Validation:
        - All old fields are required (current state from client)
        - New fields are optional (only what changed)
        - Decrypt old key to compare with new key
        - Validate provider and model name combination
        - Test effective key against effective model/provider
        - Only update fields that actually changed
        """
        try:
            if settings.ENCRYPTION_KEY is None:
                raise ValueError("Encryption key not set.")

            # Decrypt old key for comparison
            old_raw_api_key = self.decrypt_api_key(
                encryption_key=settings.ENCRYPTION_KEY,
                encrypted_key_str=payload.old_encrypted_api_key
            )

            # Determine effective values — use new if provided, else fall back to old
            effective_model_name = payload.new_embedding_model_name if payload.new_embedding_model_name else payload.old_embedding_model_name
            effective_provider = payload.new_provider if payload.new_provider else payload.old_provider
            effective_raw_key = payload.new_raw_api_key.strip() if payload.new_raw_api_key else old_raw_api_key

            # Check if anything actually changed
            nothing_changed = (
                effective_raw_key == old_raw_api_key
                and effective_model_name == payload.old_embedding_model_name
                and effective_provider == payload.old_provider
            )
            if nothing_changed:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No changes detected. Update skipped."
                )

            # Validate provider
            if not embedder_provider_validator(effective_provider):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Embedder model provider not supported."
                )

            # Validate provider and model name combination
            if not embedder_provider_mapper(effective_model_name, effective_provider):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Embedder model name and provider combination not supported."
                )

            # Test effective key against effective model and provider
            await self.test_embedding_model_api_key(
                api_key=effective_raw_key,
                model_name=effective_model_name,
                provider=effective_provider
            )

            # Build update fields — only include what actually changed
            fields_to_update = {}

            if effective_raw_key != old_raw_api_key:
                fields_to_update["api_key_encrypted"] = self.encrypt_api_key(
                    encryption_key=settings.ENCRYPTION_KEY,
                    api_key_string=effective_raw_key
                )

            if effective_model_name != payload.old_embedding_model_name:
                fields_to_update["embedding_model_name"] = effective_model_name

            if effective_provider != payload.old_provider:
                fields_to_update["provider"] = effective_provider

            if not fields_to_update:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No valid fields provided for Embedding Model API key update."
                )

            # Patch
            embedding_model_key: EmbeddingModelKey = await self.chatbot_repo.patch_embedding_model_key(
                project_id=payload.project_id,
                payload=fields_to_update
            )

            if embedding_model_key is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Embedding Model API key failed to update."
                )

            # Invalidate redis cache
            await redis_service.invalidate_chatbot_config_data_cache(
                key=embedding_model_key.chatbot_id,
                prefix=self.cached_prefix
            )
            
            all_chunk_configs: list[dict] | None = None
            
            # if provider and embedding_model_name changes, reembed
            if fields_to_update.get("embedding_model_name") or fields_to_update.get("provider"):
                # select only necessary rows. pass as payload for worker
                logger.info('[INFO] Model name and provider updated. REEMBED DOCUMENTS AGAIN')
                all_chunk_configs: list[dict] | None = await self.chunk_config_repo.get_all_configs_by_chatbot_id(embedding_model_key.chatbot_id)
                logger.info(f'[INFO]REEMBEDDING WILL{' NOT ' if not all_chunk_configs else ' '}HAPPEN')
            
            return ResponseEmbbedingModelSchema(
                id=embedding_model_key.id,
                user_id=embedding_model_key.user_id,
                chatbot_id=embedding_model_key.chatbot_id,
                api_key=embedding_model_key.api_key_encrypted,
                embedding_model_name=embedding_model_key.embedding_model_name,
                provider=embedding_model_key.provider,
                created_at=embedding_model_key.created_at,
                updated_at=embedding_model_key.updated_at
            ), all_chunk_configs

        except HTTPException:
            raise
        except Exception as e:
            await self.db.rollback()
            raise e
        
        
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    def encrypt_api_key(self, encryption_key: str, api_key_string: str) -> bytes:
        """Encrypts an API key string and returns encrypted string."""
        cipher_suite = Fernet(encryption_key.encode())
        return cipher_suite.encrypt(api_key_string.encode()).decode()

    def decrypt_api_key(self, encryption_key: str, encrypted_key_str: str) -> str:
        """Decrypts encrypted bytes and returns original string."""
        cipher_suite = Fernet(encryption_key.encode())
        return cipher_suite.decrypt(encrypted_key_str.encode()).decode()

    async def test_embedding_model_api_key(
        self, api_key: str, model_name: str, provider: str,
    ):
        """
        Test Embedding model key if working
        """
        try:
            retriever = RetrieverFactory.create_retrieval(
                provider=provider, api_key=api_key,
                model_name=model_name, qdrant=self.qdrant
            )
            await retriever.test_retrieve_embeddings()
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        except (EmbedderAuthError, EmbedderModelNotFoundError, 
                EmbedderRateLimitError, EmbedderConnectionError) as e:
            BaseRetriever.raise_http_from_retrieval_error(e)
        