from fastapi import status
from uuid import UUID
from fastapi import HTTPException
from sqlalchemy import and_, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.base.base_crud_repository import BaseCrudRepository
from api.models.chatbot import Chatbot
from api.models.document import Document
from api.models.project_member import ProjectMember
from api.models.project import Project


class ProjectRepository(BaseCrudRepository[Project]):
    """
    Project Repository for direct DB operations.
    
    Inherits standardized CRUD from BaseCrudRepository.
    """
    def __init__(self, db: AsyncSession) -> None:
        super().__init__(Project, db)
        
    async def get_project_with_member(self, project_id: UUID, user_id: UUID):
        try:
            stmt = (
                select(Project, ProjectMember.role)
                .outerjoin(
                    ProjectMember,
                    and_(
                        Project.id == ProjectMember.project_id,
                        ProjectMember.user_id == user_id
                    )
                )
                .where(Project.id == project_id)
            )

            result = await self.db.execute(stmt)
            return result.first()

        except Exception as e:
            self.db.rollback()
            raise e
    
    
    async def leave_project(
        self, project_id: UUID, current_user_id: UUID
    ) -> tuple[bool, UUID | None]:
        """
        Leave project
        Validation using Join and Where:
        - user must be a current member
        - if owner_count == 1 and total_count > 1: raise (must transfer ownership first)
        - if owner is the only member: delete entire project and return chatbot_id for cleanup
        """
        try:
            # Verify project exists
            project = await self.get_by_id(id=project_id)
            if project is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Project not found"
                )

            # Check membership + counts in one query
            owner_count_sub = (
                select(func.count(ProjectMember.user_id))
                .where(and_(
                    ProjectMember.project_id == project_id,
                    ProjectMember.role == 'owner'
                ))
                .scalar_subquery()
            )
            total_count_sub = (
                select(func.count(ProjectMember.user_id))
                .where(ProjectMember.project_id == project_id)
                .scalar_subquery()
            )

            stmt = (
                select(
                    ProjectMember,
                    owner_count_sub.label("owner_count"),
                    total_count_sub.label("total_count")
                )
                .where(and_(
                    ProjectMember.project_id == project_id,
                    ProjectMember.user_id == current_user_id
                ))
            )

            result = await self.db.execute(stmt)
            row = result.first()

            if row is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="You are not a member of this project"
                )

            target, owner_count, total_count = row

            # Block if sole owner with other members
            if target.role == 'owner' and owner_count == 1 and total_count > 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="You are the only owner. Transfer ownership before leaving."
                )

            chatbot_id = None

            # If sole member (owner leaving alone) → delete entire project
            if total_count <= 1:
                # Fetch chatbot_id before cascade delete
                # outerjoin Document since chatbot may have no documents yet
                chatbot_stmt = (
                    select(Chatbot.id)
                    .where(Chatbot.project_id == project_id)
                )
                chatbot_result = await self.db.execute(chatbot_stmt)
                chatbot_row = chatbot_result.first()
                chatbot_id = chatbot_row[0] if chatbot_row else None

                # Delete project — cascades to members, chatbots, documents via FK
                await self.db.execute(
                    delete(Project).where(Project.id == project_id)
                )

            else:
                # Just remove this member
                delete_stmt = (
                    delete(ProjectMember)
                    .where(and_(
                        ProjectMember.project_id == project_id,
                        ProjectMember.user_id == current_user_id
                    ))
                )

                await self.db.execute(delete_stmt)

            await self.db.commit()
            return True, chatbot_id

        except HTTPException:
            raise
        except Exception as e:
            await self.db.rollback()
            raise e
    
    
    async def delete_project(
        self, project_id: UUID, owner_id: UUID
    ) -> tuple[bool, UUID | None]:
        """
        Delete owner project cascades all and delete all records in GCS, QDrant
        return chatbot_id (if there is), [document.storage_path] (if there is)
        """
        try:
            # Verify project exists and user is the owner
            stmt = (
                select(Project)
                .where(and_(
                    Project.id == project_id,
                    Project.owner_id == owner_id
                ))
            )
            result = await self.db.execute(stmt)
            project = result.scalar_one_or_none()

            if project is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Project not found or you are not the owner."
                )

            # Fetch chatbot_id before cascade delete
            chatbot_stmt = (
                select(Chatbot.id)
                .where(Chatbot.project_id == project_id)
            )
            chatbot_result = await self.db.execute(chatbot_stmt)
            chatbot_row = chatbot_result.first()
            chatbot_id = chatbot_row[0] if chatbot_row else None

            # Delete project — cascades to members, chatbots, documents via FK
            await self.db.execute(
                delete(Project).where(Project.id == project_id)
            )

            await self.db.commit()
            return True, chatbot_id

        except HTTPException:
            raise
        except Exception:
            await self.db.rollback()
            raise
