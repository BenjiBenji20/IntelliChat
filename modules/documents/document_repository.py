from uuid import UUID
from typing import Optional

from fastapi import HTTPException
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from base.base_crud_repository import BaseCrudRepository
from models.document import Document


class DocumentRepository(BaseCrudRepository[Document]):
    """
    Document Repository for direct DB operations.
    
    Inherits standardized CRUD from BaseCrudRepository.
    """
    def __init__(self, db: AsyncSession) -> None:
        super().__init__(Document, db)
        self.db = db
        
    
    # ------------------------------------------------------------------
    # Bulk insert — one round trip for N documents
    # ------------------------------------------------------------------
    async def bulk_create_documents(
        self,
        *,
        user_id: UUID,
        chatbot_id: UUID,
        files: list[dict],  # [{"file_name": ..., "file_type": ..., "storage_path": "", "status": "pending"}]
    ) -> list[Document]:
        documents = [
            Document(
                user_id=user_id,
                chatbot_id=chatbot_id,
                file_name=f["file_name"],
                file_type=f["file_type"],
                file_size=f["file_size"],
                storage_path=f["storage_path"],
                status=f.get("status", "pending"),
            )
            for f in files
        ]
        self.db.add_all(documents)
        await self.db.flush()

        # Refresh all to populate server-side defaults (id, created_at, etc.)
        for doc in documents:
            await self.db.refresh(doc)

        return documents


    # ------------------------------------------------------------------
    # Bulk status update — one UPDATE ... WHERE id IN (...) round trip
    # ------------------------------------------------------------------
    async def bulk_update_status(
        self, document_ids: list[UUID], status: str
    ) -> None:
        try:
            await self.db.execute(
                update(Document)
                .where(Document.id.in_(document_ids))
                .values(status=status)
            )
            await self.db.flush()
        
        except HTTPException:
            raise
        
        except Exception as e:
            raise e


    # ------------------------------------------------------------------
    # Paginated list — never return all rows unbounded
    # ------------------------------------------------------------------
    async def get_all_by_chatbot_id(
        self,
        chatbot_id: UUID,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[Document], int]:
        try:
            # Total count — separate scalar query, cheap with the existing index
            count_result = await self.db.execute(
                select(func.count())
                .select_from(Document)
                .where(Document.chatbot_id == chatbot_id)
            )
            total = count_result.scalar_one()

            # Paginated rows
            result = await self.db.execute(
                select(Document)
                .where(Document.chatbot_id == chatbot_id)
                .order_by(Document.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            
            return list(result.scalars().all()), total
        
        except HTTPException:
            raise
        
        except Exception as e:
            raise e


    # ------------------------------------------------------------------
    # Fetch by IDs scoped to chatbot — single IN query
    # ------------------------------------------------------------------
    async def get_by_ids_and_chatbot_id(
        self, document_ids: list[UUID], chatbot_id: UUID
    ) -> list[Document]:
        try:
            result = await self.db.execute(
                select(Document).where(
                    Document.id.in_(document_ids),
                    Document.chatbot_id == chatbot_id,
                )
            )
            return list(result.scalars().all())
        
        except HTTPException:
            raise
        
        except Exception as e:
            raise e
        

    # ------------------------------------------------------------------
    # GET specific document by document id
    # ------------------------------------------------------------------
    async def get_by_document_and_chatbot_id(
        self, document_id: UUID, chatbot_id: UUID
    ) -> Optional[Document]:
        """
        Always scope fetches to chatbot_id to prevent cross-project access.
        """
        try:
            result = await self.db.execute(
                select(Document)
                .where(
                    Document.id == document_id,
                    Document.chatbot_id == chatbot_id,
                )
            )
            return result.scalar_one_or_none()
        
        except HTTPException:
            raise
        
        except Exception as e:
            raise e