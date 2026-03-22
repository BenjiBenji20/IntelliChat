from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID
import json

from api.models.chatbot import Chatbot
from api.modules.chatbot.chatbot_schema import *
from api.modules.chatbot.chatbot_repository import ChatbotRepository
from api.modules.cache.redis_service import redis_service, FREQ_CACHE_PREFIX, FREQ_CACHE_TTL


class ChatbotService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.cache_prefix=f"{FREQ_CACHE_PREFIX}(chatbot_current_state)"
        self.chatbot_repo = ChatbotRepository(db)

    async def chatbot_current_state(self, project_id: UUID) -> ChatbotStateSchema:
        """
        Check the current chatbot state
        This will be use for persistent chatbot configuration
        
        Response:
        "identity_completed": true/false,
        "llm_completed": true/false,
        "embedding_completed": true/false
        """
        try:
            # holds None value if not hit
            cached_state = await redis_service.get_hash(key=str(project_id), prefix=self.cache_prefix)
            
            if cached_state:
                # Deserialize string values back to correct types
                state = {
                    "chatbot_id": json.loads(cached_state.get("chatbot_id")),
                    "chatbot_completed": json.loads(cached_state.get("chatbot_completed")),
                    "llm_completed": json.loads(cached_state.get("llm_completed")),
                    "embedding_completed": json.loads(cached_state.get("embedding_completed")),
                    "chatbot_data": json.loads(cached_state.get("chatbot_data")),
                    "llm_data": json.loads(cached_state.get("llm_data")),
                    "embedding_data": json.loads(cached_state.get("embedding_data")),
                }
            else:
                state = await self.chatbot_repo.get_chatbot_setup_status(project_id)
                if state is None:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Project not found."
                    )
            
            # cache state in redis
            if not cached_state:
                attempts = 0
                is_cached = False
                while not is_cached and attempts < 3: # make 3 attempts
                    is_cached = await redis_service.set_nested_dict_hash(
                        key=str(project_id), prefix=self.cache_prefix,
                        data=state, ttl=FREQ_CACHE_TTL # 12hrs
                    )
                    attempts += 1
                # delete key on update data
                
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
                
            # invalidate chatbot current state cache
            await redis_service.invalidate_chatbot_config_data_cache(
                key=str(payload_dict["project_id"]), prefix=self.cache_prefix
            )
            
            return ResponseChatbotSchema(
                id=chatbot.id,
                user_id=chatbot.user_id,
                project_id=chatbot.project_id,
                application_name=chatbot.application_name,
                has_memory=chatbot.has_memory,
                is_active=chatbot.is_active,
                created_at=chatbot.created_at,
                updated_at=chatbot.updated_at
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
                
            # delete cached chatbot current state data
            # does not invalidate cached api_key_(chatbot_config_data) since its just holding api keys
            await redis_service.invalidate_chatbot_config_data_cache(
                key=project_id, prefix=self.cache_prefix
            )
            
            return ResponseChatbotSchema(
                id=chatbot.id,
                user_id=chatbot.user_id,
                project_id=chatbot.project_id,
                application_name=chatbot.application_name,
                has_memory=chatbot.has_memory,
                is_active=chatbot.is_active,
                created_at=chatbot.created_at,
                updated_at=chatbot.updated_at
            )
            
        except HTTPException:
            raise
        except Exception as e:
            await self.db.rollback()
            raise e
  