from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException
from sqlalchemy import select
from uuid import UUID

from api.base.base_crud_repository import BaseCrudRepository
from api.models.chunking_configuration import ChunkingConfiguration
from api.models.chatbot import Chatbot

from logging import getLogger

from api.models.document import Document

logger = getLogger(__name__)

class ChunkingConfigurationRepository(BaseCrudRepository[ChunkingConfiguration]):
    """
    ChunkingConfiguration Repository for direct DB operations.
    
    Inherits standardized CRUD from BaseCrudRepository.
    """
    def __init__(self, db: AsyncSession) -> None:
        super().__init__(ChunkingConfiguration, db)
        self.db = db
        
        
    async def get_all_configs_by_chatbot_id(self, chatbot_id: UUID) -> list[dict] | None:
        try:
            stmt = (
                select(
                    Document.chatbot_id,
                    Document.id.label("document_id"),
                    Document.file_name,
                    Document.file_type,
                    ChunkingConfiguration.chunk_size,
                    ChunkingConfiguration.chunk_overlap,
                    ChunkingConfiguration.separator,
                    ChunkingConfiguration.document_type,
                )
                .join(
                    ChunkingConfiguration,
                    ChunkingConfiguration.document_id == Document.id
                )
                .where(ChunkingConfiguration.chatbot_id == chatbot_id)
            )

            result = await self.db.execute(stmt)
            rows = result.mappings().all()

            if not rows:
                logger.info(f"[DEBUG] No chunking configurations found for chatbot_id: {chatbot_id}")
                return None

            return [dict(row) for row in rows]

        except HTTPException:
            raise
        except Exception:
            await self.db.rollback()
            raise
        