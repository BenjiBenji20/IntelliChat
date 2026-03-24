from cryptography.fernet import Fernet, InvalidToken
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from groq import Groq
import asyncio

from api.modules.cache.redis_service import API_KEY_CACHE_PREFIX, redis_service, FREQ_CACHE_PREFIX
from api.models.llm_key import LlmKey
from api.modules.chat.llm.base_llm import *
from api.modules.chatbot.chatbot_repository import ChatbotRepository
from api.modules.llm_api_keys.llm_key_repository import LlmKeyRepository
from api.modules.embedding_model_api_keys.embedding_model_key_repository import EmbeddingModelKeyRepository
from api.modules.llm_api_keys.llm_api_keys_schema import *
from api.configs.settings import settings
from api.modules.chat.llm.llm_factory import LLMFactory
from shared.ai_models_details import llm_provider_mapper, llm_provider_validator

class ChatbotAPIKeyService:
    def __init__(self, db: AsyncSession):
        """
        This service will interact with 3 models: Chatbot, LLMKey and EmbeddingModelKey
        Chatbot model creation has steps:
        1. Chatbot's identity: User creates a chatbot name, prompt, etc...
        2. Chatbot's knowledge: User uploads API keys [LLM, Embedding Model]
        """
        self.db = db
        self.chatbot_repo = ChatbotRepository(db)
        self.llm_key_repo = LlmKeyRepository(db)
        self.embedding_model_key_repo = EmbeddingModelKeyRepository(db)
        self.cached_prefix = f"{API_KEY_CACHE_PREFIX}(chatbot_config_data)"
    
    
    async def upload_llm_key(self, payload: CreateRequestLlmSchema) -> ResponseLlmSchema:
        """
        User inputs:
            -  Llm API Key, model name, temperatur, provider
        Table: llm_keys
        Flow:
            Test API token
            Encypt raw api key and store in payload
            Store in DB
        """
        try:
            payload_dict = payload.model_dump()
            
            raw_llm_key = payload_dict['api_key']
            llm_name = payload_dict["llm_name"]
            provider = payload_dict["provider"]
            # get and pop project_id 
            project_id = payload_dict["project_id"] # use for invalidate caching
            payload_dict.pop("project_id")
            
            if not llm_provider_validator(provider):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="LLM provider not supported."
                )
            
            if not llm_provider_mapper(llm_name, provider):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="LLM name and provider combination not supported."
                )
            
            # test new api key
            # test the model if it has token
            await self.test_llm_api_key(
                api_key=raw_llm_key,
                model_name=llm_name,
                provider=provider 
            )
            
            if settings.ENCRYPTION_KEY is None:
                raise ValueError("Encryption key not set.")
            
            payload_dict.update({'api_key_encrypted': self.encrypt_api_key(
                encryption_key=settings.ENCRYPTION_KEY,
                api_key_string=raw_llm_key
            )})
            
            payload_dict.pop("api_key") # remove the raw api_key
            
            llm_key: LlmKey = await self.llm_key_repo.create(**payload_dict)
            
            if not llm_key:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to register API key. Please try again."
                )
                
            # invalidate chatbot current state cache
            await redis_service.delete(
                key=str(project_id), prefix=f"{FREQ_CACHE_PREFIX}(chatbot_current_state)"
            )
        
            return ResponseLlmSchema(
                id=llm_key.id,
                user_id=llm_key.user_id,
                chatbot_id=llm_key.chatbot_id,
                project_id=project_id,
                api_key=llm_key.api_key_encrypted,
                llm_name=llm_key.llm_name,
                temperature=llm_key.temperature,
                provider=llm_key.provider,
                created_at=llm_key.created_at,
                updated_at=llm_key.updated_at, 
            )
        
        except HTTPException:
            raise
        except Exception as e:
            await self.db.rollback()
            raise e    
    
  
    async def update_llm_api_key(
        self, payload: UpdateRequestLlmSchema
    ) -> ResponseLlmSchema:
        """
        Update: llm_keys
        Not to update: id, user_id, created_at and original chatbot_id
        
        Validation:
        - All old fields are required (current state from client)
        - New fields are optional (only what changed)
        - Decrypt old key to compare with new key
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
            effective_llm_name = payload.new_llm_name.lower().strip() if payload.new_llm_name else payload.old_llm_name
            effective_provider = payload.new_provider.lower().strip() if payload.new_provider else payload.old_provider
            effective_temperature = payload.new_temperature if payload.new_temperature is not None else payload.old_temperature
            effective_raw_key = payload.new_raw_api_key.strip() if payload.new_raw_api_key else old_raw_api_key

            # Check if anything actually changed
            nothing_changed = (
                effective_raw_key == old_raw_api_key
                and effective_llm_name == payload.old_llm_name
                and effective_provider == payload.old_provider
                and effective_temperature == payload.old_temperature
            )
            if nothing_changed:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No changes detected. Update skipped."
                )

            # Validate provider
            if not llm_provider_validator(effective_provider):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Provider not supported."
                )

            # Validate provider and llm name are matched
            if not llm_provider_mapper(effective_llm_name, effective_provider):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="LLM name and provider combination not supported."
                )

            # Test effective key against effective model and provider
            await self.test_llm_api_key(
                api_key=effective_raw_key,
                model_name=effective_llm_name,
                provider=effective_provider
            )

            # Build update fields — only include what actually changed
            fields_to_update = {}

            if effective_raw_key != old_raw_api_key:
                fields_to_update["api_key_encrypted"] = self.encrypt_api_key(
                    encryption_key=settings.ENCRYPTION_KEY,
                    api_key_string=effective_raw_key
                )

            if effective_llm_name != payload.old_llm_name:
                fields_to_update["llm_name"] = effective_llm_name

            if effective_provider != payload.old_provider:
                fields_to_update["provider"] = effective_provider

            if effective_temperature != payload.old_temperature:
                fields_to_update["temperature"] = effective_temperature

            if not fields_to_update:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No valid fields provided for LLM API key update."
                )

            # Patch
            llm_key: LlmKey = await self.chatbot_repo.patch_llm_key(
                project_id=payload.project_id,
                payload=fields_to_update
            )

            if llm_key is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="LLM API key failed to update."
                )

            # Invalidate redis cache
            await redis_service.delete(
                key=str(llm_key.chatbot_id),
                prefix=self.cached_prefix
            )

            return ResponseLlmSchema(
                id=llm_key.id,
                user_id=llm_key.user_id,
                chatbot_id=llm_key.chatbot_id,
                project_id=payload.project_id,
                api_key=llm_key.api_key_encrypted,
                llm_name=llm_key.llm_name,
                temperature=llm_key.temperature,
                provider=llm_key.provider,
                created_at=llm_key.created_at,
                updated_at=llm_key.updated_at,
            )

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
        try:
            cipher_suite = Fernet(encryption_key.encode())
            return cipher_suite.decrypt(encrypted_key_str.encode()).decode()
        except InvalidToken:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Encrypted key is invalid or corrupted. Please re-enter your API key."
            )
    
    async def test_llm_api_key(
        self,
        api_key: str,
        model_name: str,
        provider: str
    ):
        """
        Test LLM if working
        """
        try:
            llm = LLMFactory.create_llm(
                api_key=api_key,
                model_name=model_name,
                provider=provider
            )
            
            await llm.test_llm()
        except ValueError as e:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
            
        except (LLMAuthError, LLMModelNotFoundError, LLMRateLimitError, LLMConnectionError) as e:
            BaseLLM.raise_http_from_llm_error(e)
        