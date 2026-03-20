from uuid import UUID

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from api.base.base_crud_repository import BaseCrudRepository
from api.models.embedding_model_key import EmbeddingModelKey
from api.models.chatbot import Chatbot


class EmbeddingModelKeyRepository(BaseCrudRepository[EmbeddingModelKey]):
    """
    EmbeddingModelKey Repository for direct DB operations.
    
    Inherits standardized CRUD from BaseCrudRepository.
    """
    def __init__(self, db: AsyncSession) -> None:
        super().__init__(EmbeddingModelKey, db)
        
        
    async def get_embedding_model_details(self, chatbot_id: UUID) -> dict:
        """
        Fetch model details necessary for retrieval.
        """
        try:
            
            stmt = (
                select(
                    EmbeddingModelKey.id,
                    EmbeddingModelKey.api_key_encrypted,
                    EmbeddingModelKey.embedding_model_name,
                    EmbeddingModelKey.provider.label("embedding_provider")
                )
                .join(
                    Chatbot, Chatbot.id == EmbeddingModelKey.chatbot_id
                )
                .where(
                    Chatbot.id == chatbot_id
                )
            )
            
            result = await self.db.execute(stmt)
            
            return result.mappings().first()
            
        except HTTPException:
            raise
        
        except Exception as e:
            await self.db.rollback()
            raise e
        