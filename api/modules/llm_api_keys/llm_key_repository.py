from uuid import UUID

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from api.base.base_crud_repository import BaseCrudRepository
from api.models.chatbot import Chatbot
from api.models.llm_key import LlmKey


class LlmKeyRepository(BaseCrudRepository[LlmKey]):
    """
    LlmKey Repository for direct DB operations.
    
    Inherits standardized CRUD from BaseCrudRepository.
    """
    def __init__(self, db: AsyncSession) -> None:
        super().__init__(LlmKey, db)
    
    
    async def get_llm_details(self, chatbot_id: UUID) -> dict | None:
        try:
            stmt = (
                select(
                    LlmKey.api_key_encrypted,
                    LlmKey.llm_name,
                    LlmKey.provider.label("llm_provider"),
                    LlmKey.temperature
                )
                .join(Chatbot, Chatbot.id == LlmKey.chatbot_id)
                .where(Chatbot.id == chatbot_id)
            )
            
            result = await self.db.execute(stmt)
            return result.mappings().one_or_none()
                        
        except HTTPException:
            raise
        
        except Exception:
            await self.db.rollback()
            raise
    