from typing import Optional
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import select

from base.base_crud_repository import BaseCrudRepository
from models.profile import Profile
from models.project import Project

from sqlalchemy.ext.asyncio import AsyncSession

from models.project_member import ProjectMember


class ProjectRepository(BaseCrudRepository[Project]):
    """
    Project Repository for direct DB operations.
    
    Inherits standardized CRUD from BaseCrudRepository.
    """
    def __init__(self, db: AsyncSession) -> None:
        super().__init__(Project, db)
        
    
    async def get_project_member_by_id(self, user_id: UUID, project_id: UUID) -> Optional[Profile]:
        try:
            stmt = (
                select(Profile)
                .join(ProjectMember, ProjectMember.project_id == project_id)
                .where(
                    ProjectMember.user_id == user_id
                )
            )
            
            results = await self.db.execute(stmt)
            return results.scalar_one_or_none()
            
        except HTTPException:
            raise
        except Exception as e:
            await self.db.rollback()
            raise e
