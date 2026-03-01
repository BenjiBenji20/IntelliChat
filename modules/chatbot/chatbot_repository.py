from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID
from sqlalchemy import and_, select, case

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