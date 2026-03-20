from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.models.chatbot_behavior import ChatbotBehavior
from api.base.base_crud_repository import BaseCrudRepository


class ChatbotBehaviorRepository(BaseCrudRepository[ChatbotBehavior]):
    """
    ChatbotBehavior Repository for direct DB operations.
    
    Inherits standardized CRUD from BaseCrudRepository.
    """
    def __init__(self, db: AsyncSession) -> None:
        super().__init__(ChatbotBehavior, db)


    async def patch_chatbot_behavior(self, chatbot_id: UUID, payload: dict) -> ChatbotBehavior | None:
        """
        Partially update ChatbotBehavior
        Not to update: id, user_id, created_at and original chatbot_id
        """
        try:
            stmt = (
                select(ChatbotBehavior)
                .where(ChatbotBehavior.chatbot_id == chatbot_id)
            )
            
            result = await self.db.execute(stmt)
            behavior: ChatbotBehavior | None = result.scalar_one_or_none()
            
            if behavior is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Chatbot's behavior not found."
                )
            
            # Apply updates to the model instance
            for field, value in payload.items():
                setattr(behavior, field, value)

            await self.db.commit()
            await self.db.refresh(behavior)

            return behavior
            
        except HTTPException:
            raise
        except Exception as e:
            await self.db.rollback()
            raise e
    
    
    async def get_system_prompt(self, chatbot_id: UUID) -> str | None:
        try:
            stmt = (
                select(ChatbotBehavior.system_prompt)
                .where(ChatbotBehavior.chatbot_id == chatbot_id)
            )
            
            result = await self.db.execute(stmt)
            return result.scalar()
            
        except Exception:
            self.db.rollback()
            raise
          