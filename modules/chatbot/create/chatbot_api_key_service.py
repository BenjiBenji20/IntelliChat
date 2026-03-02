from cryptography.fernet import Fernet
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from groq import Groq
import asyncio
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from models.chatbot import Chatbot
from models.llm_key import LlmKey
from models.embedding_model_key import EmbeddingModelKey
from modules.chatbot.chatbot_repository import ChatbotRepository
from modules.chatbot.llm_key_repository import LlmKeyRepository
from modules.chatbot.embedding_model_key_repository import EmbeddingModelKeyRepository
from schemas.chatbot_schema import *
from configs.settings import settings

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
    
    async def check_chatbot_step(self, project_id: UUID) -> ChatbotStateSchema:
        """
        Check the current chatbot state
        This will be use for persistent chatbot configuration
        
        Response:
        "identity_completed": true/false,
        "llm_completed": true/false,
        "embedding_completed": true/false
        """
        try:
            state = await self.chatbot_repo.get_chatbot_setup_status(project_id)
            
            if state is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Project not found."
                )

            return ChatbotStateSchema(
                chatbot_id=state["chatbot_id"],
                chatbot_completed=state["chatbot_completed"],
                llm_completed=state["llm_completed"],
                embedding_completed=state["embedding_completed"],
                chatbot_data=state.get("chatbot_data"),
                llm_data=state.get("llm_data"),
                embedding_data=state.get("embedding_data")
            )
            
        except HTTPException:
            raise
        except Exception as e:
            await self.db.rollback()
            raise e   
    
    
    async def create_chatbot_identity(self, payload: CreateRequestChatbotSchema) -> ResponseChatbotSchema:
        """
        User inputs:
            - chatbot name, prompt, etc...
        Table: chatbots
        """
        try:
            payload_dict = payload.model_dump()
            chatbot: Chatbot = await self.chatbot_repo.create(**payload_dict)
            
            if chatbot is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Chatbot {payload_dict["application_name"]} failed to create."
                )
                
            return ResponseChatbotSchema(
                id=chatbot.id,
                user_id=chatbot.user_id,
                project_id=chatbot.project_id,
                application_name=chatbot.application_name,
                has_memory=chatbot.has_memory,
                system_prompt=chatbot.system_prompt,
                is_active=chatbot.is_active,
                created_at=chatbot.created_at,
                updated_at=chatbot.updated_at
            )
        
        except HTTPException:
            raise
        except Exception as e:
            await self.db.rollback()
            raise e
    
    
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
            
            try:
                # test the model if it has token
                await asyncio.to_thread(
                    self.test_llm_api_key,
                    raw_llm_key,
                    payload_dict["llm_name"],
                    payload_dict.get("temperature", 0.70)
                )

            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="LLM API key is invalid or failed to authenticate."
                )
            
            if settings.ENCRYPTION_KEY is None:
                raise ValueError("Encryption key not set.")
            
            payload_dict.update({'api_key_encrypted': self.encrypt_api_key(
                encryption_key=settings.ENCRYPTION_KEY,
                api_key_string=raw_llm_key
            )})
            
            payload_dict.pop("api_key") # remove the raw api_key
            
            llm_key: LlmKey = await self.llm_key_repo.create(**payload_dict)
        
            return ResponseLlmSchema(
                id=llm_key.id,
                user_id=llm_key.user_id,
                chatbot_id=llm_key.chatbot_id,
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
    

    async def update_chatbot_identity(
        self, payload: UpdateRequestChatbotSchema
    ) -> ResponseChatbotSchema:
        """
        Not to update: id, user_id, created_at and original project_id
        """
        try:
            payload_dict = payload.model_dump(exclude_unset=True) # only fields client actually sent
            project_id = payload_dict["project_id"]
            
            # Strip out protected fields from payload
            protected_fields = {"id", "user_id", "project_id", "created_at"}
            update_data = {k: v for k, v in payload_dict.items() if k not in protected_fields}
                        
            if not update_data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No valid fields provided for Chatbot update."
                )
                
            chatbot: Chatbot = await self.chatbot_repo.patch_chatbot_identity(
                project_id=project_id, payload=update_data
            )
            
            if chatbot is None:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail=f"Chatbot {update_data["application_name"]} failed to update."
                )
                
            return ResponseChatbotSchema(
                id=chatbot.id,
                user_id=chatbot.user_id,
                project_id=chatbot.project_id,
                application_name=chatbot.application_name,
                has_memory=chatbot.has_memory,
                system_prompt=chatbot.system_prompt,
                is_active=chatbot.is_active,
                created_at=chatbot.created_at,
                updated_at=chatbot.updated_at
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
        Pass API key with model name undergoes minimal test call
        Broken API key wont reach the repo
        """
        try:
            payload_dict = payload.model_dump(exclude_unset=True) # only fields client actually sent
            raw_llm_key = payload_dict.get("api_key", None)
            
            if raw_llm_key:
                # test new api key
                try:
                    # test the model if it has token
                    await asyncio.to_thread(
                        self.test_llm_api_key,
                        raw_llm_key,
                        payload_dict["llm_name"],
                        payload_dict.get("temperature", 0.70)
                    )
    
                except Exception:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Your new API key is broken. Failed update."
                    )
                
                if settings.ENCRYPTION_KEY is None:
                    raise ValueError("Encryption key not set.")
                
                # encrypt updated api key if there is 
                if payload_dict["api_key"]:
                    payload_dict.update({'api_key_encrypted': self.encrypt_api_key(
                        encryption_key=settings.ENCRYPTION_KEY,
                        api_key_string=raw_llm_key
                    )})
                    payload_dict.pop("api_key") # remove the raw api_key
            
            project_id = payload_dict["project_id"]
            
            # Strip out protected fields from payload
            protected_fields = {"id", "user_id", "chatbot_id", "created_at"}
            update_data = {k: v for k, v in payload_dict.items() if k not in protected_fields}
                        
            if not update_data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No valid fields provided for LLM API key update."
                )
                
            llm_key: LlmKey = await self.chatbot_repo.patch_llm_key(
                project_id=project_id, payload=update_data
            )
            
            if llm_key is None:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail=f"LLM API key failed to update."
                )
                
            return ResponseLlmSchema(
                id=llm_key.id,
                user_id=llm_key.user_id,
                chatbot_id=llm_key.chatbot_id,
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
    
    def test_llm_api_key(
        self,
        api_key: str,
        model_name: str = "openai/gpt-oss-120b", 
        temperature: float = 0.70,
    ) -> str | None:
        """
        Test LLM if working
        """
        llm = Groq(api_key=api_key)
        
        test_llm = llm.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Hello World!"}],
            temperature=temperature,
            max_tokens=1024,
            stream=True
        )
        
        if test_llm is None:
            return None
        
        # concat chunks as they arrive
        response = ""
        for reply in test_llm:
            if reply.choices[0].delta.content:
                response += reply.choices[0].delta.content
        return response

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
        