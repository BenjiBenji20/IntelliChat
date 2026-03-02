from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID
from sqlalchemy import select, case

from base.base_crud_repository import BaseCrudRepository
from models.chatbot import Chatbot
from models.embedding_model_key import EmbeddingModelKey
from models.llm_key import LlmKey


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
                select(
                    Chatbot.id.label("chatbot_id"),
                    case(
                        (Chatbot.id != None, True),
                        else_=False
                    ).label("chatbot_completed"),
                    case(
                        (LlmKey.id != None, True),
                        else_=False
                    ).label("llm_completed"),
                    case(
                        (EmbeddingModelKey.id != None, True),
                        else_=False
                    ).label("embedding_completed")
                )
                .select_from(Chatbot)
                .outerjoin(LlmKey, LlmKey.chatbot_id == Chatbot.id)
                .outerjoin(EmbeddingModelKey, EmbeddingModelKey.chatbot_id == Chatbot.id)
                .where(Chatbot.project_id == project_id)
            )

            result = await self.db.execute(stmt)
            row = result.first()

            if row is None:
                return {
                    "chatbot_id": None,
                    "chatbot_completed": False,
                    "llm_completed": False,
                    "embedding_completed": False
                }

            return {
                "chatbot_id": row.chatbot_id,
                "chatbot_completed": row.chatbot_completed,
                "llm_completed": row.llm_completed,
                "embedding_completed": row.embedding_completed
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
        