from cryptography.fernet import Fernet
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from models.chatbot import Chatbot
from models.embedding_model_key import EmbeddingModelKey
from modules.chatbot.chatbot_repository import ChatbotRepository
from modules.llm_api_keys.llm_key_repository import LlmKeyRepository
from modules.embedding_model_api_keys.embedding_model_key_repository import EmbeddingModelKeyRepository
from modules.embedding_model_api_keys.embedding_model_api_keys_schema import *
from configs.settings import settings

class EmbeddingModelAPIKeyService:
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
    
    async def upload_embedding_model_key(self, payload: CreateRequestEmbbedingModelSchema) -> ResponseEmbbedingModelSchema:
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
            
            # test the model if it has token
            try:
                test_embedding_model = await asyncio.to_thread(
                    self.test_embedding_model_api_key,
                    raw_embedding_model_key,
                    payload_dict["embedding_model_name"],
                )
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Embedding Model API key is invalid or failed to authenticate."
                )
            
            if test_embedding_model["status"] != "success":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Embedding Model API key failed: empty embedding returned."
                )
            
            if settings.ENCRYPTION_KEY is None:
                raise ValueError("Encryption key not set.")
            
            payload_dict.update({'api_key_encrypted': self.encrypt_api_key(
                encryption_key=settings.ENCRYPTION_KEY,
                api_key_string=raw_embedding_model_key
            )})
            
            payload_dict.pop("api_key") # remove the raw api_key
            
            embedding_model_key: EmbeddingModelKey = await self.embedding_model_key_repo.create(**payload_dict)
        
            return ResponseEmbbedingModelSchema(
                id=embedding_model_key.id,
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
    ) -> ResponseEmbbedingModelSchema:
        """
        Update: embedding_model_keys
        Not to update: id, user_id, created_at and original chatbot_id
        
        Validation:
        Pass API key with model name undergoes minimal test call
        Broken API key wont reach the repo
        """
        try:
            payload_dict = payload.model_dump(exclude_unset=True) # only fields client actually sent
            raw_embedding_model_key = payload_dict.get("api_key", None)
            
            if raw_embedding_model_key:
                # test new api key
                try:
                    test_embedding_model = await asyncio.to_thread(
                        self.test_embedding_model_api_key,
                        raw_embedding_model_key,
                        payload_dict["embedding_model_name"],
                    )
                except Exception:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Embedding Model API key is invalid or failed to authenticate."
                    )
                    
                if test_embedding_model["status"] != "success":
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Embedding Model API key failed: empty embedding returned."
                    )
                
                if settings.ENCRYPTION_KEY is None:
                    raise ValueError("Encryption key not set.")
                            
                # encrypt updated api key if there is 
                if payload_dict["api_key"]:
                    payload_dict.update({'api_key_encrypted': self.encrypt_api_key(
                        encryption_key=settings.ENCRYPTION_KEY,
                        api_key_string=raw_embedding_model_key
                    )})
                    payload_dict.pop("api_key") # remove the raw api_key
            
            project_id = payload_dict["project_id"]
            
            # Strip out protected fields from payload
            protected_fields = {"id", "user_id", "chatbot_id", "created_at"}
            update_data = {k: v for k, v in payload_dict.items() if k not in protected_fields}
                        
            if not update_data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No valid fields provided for Embedding Model API key update."
                )
                
            embedding_model_key: EmbeddingModelKey = await self.chatbot_repo.patch_embedding_model_key(
                project_id=project_id, payload=update_data
            )
            
            if embedding_model_key is None:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail=f"Embedding Model API key failed to update."
                )
                
            return ResponseEmbbedingModelSchema(
                id=embedding_model_key.id,
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

    def test_embedding_model_api_key(
        self,
        api_key: str,
        model_name: str = "models/gemini-embedding-001", 
    ) -> dict:
        """
        Test Embedding model key if working
        """
        embedding_model = GoogleGenerativeAIEmbeddings(
            api_key=api_key,
            model=model_name
        )
        
        # minimal test string — raises if key/model is invalid
        test_vector = embedding_model.embed_query("Hello World!")
        
        # Validate structure
        if isinstance(test_vector, list) and len(test_vector) > 0:
            return {
                "status": "success",
                "vector_dimension": len(test_vector)
            }

        return {"status": "failed", "reason": "Empty embedding returned"}
        