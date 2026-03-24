from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID
from sqlalchemy import select, case, and_

from api.base.base_crud_repository import BaseCrudRepository
from api.models.chatbot import Chatbot
from api.models.embedding_model_key import EmbeddingModelKey
from api.models.chatbot_behavior import ChatbotBehavior
from api.models.llm_key import LlmKey


class ChatbotRepository(BaseCrudRepository[Chatbot]):
    """
    Chatbot Repository for direct DB operations.
    
    Inherits standardized CRUD from BaseCrudRepository.
    """
    def __init__(self, db: AsyncSession) -> None:
        super().__init__(Chatbot, db)
    
    
    async def get_chatbot_setup_status(self, project_id: UUID):
        try:
            stmt = (
                select(Chatbot, LlmKey, EmbeddingModelKey, ChatbotBehavior.system_prompt)
                .select_from(Chatbot)
                .outerjoin(LlmKey, LlmKey.chatbot_id == Chatbot.id)
                .outerjoin(EmbeddingModelKey, EmbeddingModelKey.chatbot_id == Chatbot.id)
                .outerjoin(ChatbotBehavior, ChatbotBehavior.chatbot_id == Chatbot.id)
                .where(Chatbot.project_id == project_id)
            )

            result = await self.db.execute(stmt)
            row = result.first()

            if row is None:
                return {
                    "chatbot_id": None,
                    "chatbot_completed": False,
                    "llm_completed": False,
                    "embedding_completed": False,
                    "chatbot_data": None,
                    "llm_data": None,
                    "embedding_data": None
                }

            chatbot, llm, embedding, system_prompt = row

            return {
                "chatbot_id": chatbot.id if chatbot else None,
                "chatbot_completed": chatbot is not None,
                "llm_completed": llm is not None,
                "embedding_completed": embedding is not None,
                "system_prompt": system_prompt if not None else None,
                "chatbot_data": {
                    "id": chatbot.id,
                    "application_name": chatbot.application_name,
                    "has_memory": chatbot.has_memory
                } if chatbot else None,
                "llm_data": {
                    "id": llm.id,
                    "provider": llm.provider,
                    "llm_name": llm.llm_name,
                    "api_key_encrypted": llm.api_key_encrypted,
                    "temperature": llm.temperature
                } if llm else None,
                "embedding_data": {
                    "id": embedding.id,
                    "provider": embedding.provider,
                    "embedding_model_name": embedding.embedding_model_name,
                    "api_key_encrypted": embedding.api_key_encrypted,
                } if embedding else None
            }

        except Exception as e:
            raise e
        
    
    async def patch_chatbot_identity(self, project_id: UUID, payload: dict) -> Chatbot | None:
        """
        Partially update chatbots
        Not to update: id, user_id, created_at and original project_id
        """
        try:
            stmt = (
                select(Chatbot)
                .where(Chatbot.project_id == project_id)
            )
            
            result = await self.db.execute(stmt)
            chatbot: Chatbot | None = result.scalar_one_or_none()
            
            if chatbot is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Chatbot not found."
                )
            
            # Apply updates
            for field, value in payload.items():
                setattr(chatbot, field, value)

            await self.db.commit()
            await self.db.refresh(chatbot)

            return chatbot
            
        except HTTPException:
            raise
        except Exception as e:
            await self.db.rollback()
            raise e
        
        
    async def patch_llm_key(self, project_id: UUID, payload: dict) -> LlmKey | None:
        """
        Partially update chatbots
        Not to update: id, user_id, created_at and original chatbot_id
        """
        try:
            stmt = (
                select(LlmKey)
                .join(Chatbot, Chatbot.id == LlmKey.chatbot_id)
                .where(Chatbot.project_id == project_id)
            )
            
            result = await self.db.execute(stmt)
            llm: LlmKey | None = result.scalar_one_or_none()
            
            if llm is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"LLM API key not found."
                )
            
            # Apply updates
            for field, value in payload.items():
                setattr(llm, field, value)

            await self.db.commit()
            await self.db.refresh(llm)

            return llm
            
        except HTTPException:
            raise
        except Exception as e:
            await self.db.rollback()
            raise e
        
        
    async def patch_embedding_model_key(self, project_id: UUID, payload: dict) -> EmbeddingModelKey | None:
        """
        Partially update: embedding_model_keys
        Not to update: id, user_id, created_at and original chatbot_id
        """
        try:
            stmt = (
                select(EmbeddingModelKey)
                .join(Chatbot, Chatbot.id == EmbeddingModelKey.chatbot_id)
                .where(Chatbot.project_id == project_id)
            )
            
            result = await self.db.execute(stmt)
            embedding_model: EmbeddingModelKey | None = result.scalar_one_or_none()
            
            if embedding_model is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Embedding Model API key not found."
                )
            
            # Apply updates
            for field, value in payload.items():
                setattr(embedding_model, field, value)

            await self.db.commit()
            await self.db.refresh(embedding_model)

            return embedding_model
            
        except HTTPException:
            raise
        except Exception as e:
            await self.db.rollback()
            raise e
        

    async def get_models_identities(self, chatbot_id: UUID) -> dict:
        try:
            stmt = (
                select(
                    LlmKey.api_key_encrypted.label("llm_encrypted_key"),
                    LlmKey.llm_name,
                    LlmKey.temperature,
                    LlmKey.provider.label("llm_provider"),
                    EmbeddingModelKey.api_key_encrypted.label("embedding_encrypted_key"),
                    EmbeddingModelKey.embedding_model_name,
                    EmbeddingModelKey.provider.label("embedding_provider")
                )
                .select_from(LlmKey)
                .join(EmbeddingModelKey, EmbeddingModelKey.chatbot_id == LlmKey.chatbot_id)
                .where(LlmKey.chatbot_id == chatbot_id)
            )

            result = await self.db.execute(stmt)
            row = result.first()

            if row is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Chatbot model identity not found."
                )

            return {
                "llm_encrypted_key": row.llm_encrypted_key,
                "llm_model_name": row.llm_name,
                "llm_temperature": row.temperature,
                "llm_provider": row.llm_provider,
                "embedding_model_encrypted_key": row.embedding_encrypted_key,
                "embedding_model_name": row.embedding_model_name,
                "embedding_model_provider": row.embedding_provider
            }

        except HTTPException:
            raise
        except Exception as e:
            await self.db.rollback()
            raise e
