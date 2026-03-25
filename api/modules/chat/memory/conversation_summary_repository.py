from uuid import UUID
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from api.base.base_crud_repository import BaseCrudRepository
from api.models.conversation_summary import ConversationSummary
from sqlalchemy import and_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import func

logger = logging.getLogger(__name__)

class ConversationSummaryRepository(BaseCrudRepository[ConversationSummary]):

    """
    ConversationSummary Repository for direct DB operations.
    
    Inherits standardized CRUD from BaseCrudRepository.
    """
    def __init__(self, db: AsyncSession) -> None:
        super().__init__(ConversationSummary, db)


    async def get_summary(self, chatbot_id: UUID, session_id: str) -> str | None:
        try:
            stmt = (
                select(ConversationSummary.summary)
                .where(and_(
                    ConversationSummary.session_id == session_id,
                    ConversationSummary.chatbot_id == chatbot_id
                ))
                .order_by(ConversationSummary.created_at.desc())
                .limit(1)
            )
            
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"[ERROR] Error fetching summary from database: {e}")
            await self.db.rollback()
            return None


    # UPSERT QUERY
    async def save_summary(
        self,
        session_id: str,
        chatbot_id: UUID,
        summary: str,
        token_count: int,
    ) -> None:
        try:
            stmt = insert(ConversationSummary).values(
                session_id=session_id,
                chatbot_id=chatbot_id,
                summary=summary,
                token_count=token_count,
            )
            
            # update action when a conflict on 'session_id' occurs
            on_update_stmt = stmt.on_conflict_do_update(
                index_elements=["session_id"],
                set_={
                    "summary": stmt.excluded.summary,
                    "token_count": stmt.excluded.token_count,
                    "updated_at": func.now()
                }
            )

            await self.db.execute(on_update_stmt)
            await self.db.commit()
        except Exception as e:
            logger.error(f"[ERROR] Error storing summary to database: {e}")
            await self.db.rollback()

