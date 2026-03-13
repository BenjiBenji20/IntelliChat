from fastapi import status
from uuid import UUID
from fastapi import HTTPException
from sqlalchemy import and_, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.base.base_crud_repository import BaseCrudRepository
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
            raise e
    
    
    async def leave_project(self, project_id: UUID, current_user_id: UUID) -> bool:
        """
            Leave project
            Validation using Join and Where:
            - user must be a current user
            - if total count > 1 and owner count == 1: raise
        """
        try:
            # 1. Fetch project to check baseline ownership
            project = await self.get_by_id(id=project_id)
            if project is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Project not found"
                )

            # 2. Check membership and get counts in one go using subqueries
            owner_count_sub = (
                select(func.count(ProjectMember.user_id))
                .where(and_(ProjectMember.project_id == project_id, ProjectMember.role == 'owner'))
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
                .where(
                    and_(
                        ProjectMember.project_id == project_id,
                        ProjectMember.user_id == current_user_id
                    )
                )
            )

            result = await self.db.execute(stmt)
            row = result.first()

            # Determine ownership status (handling backward compatibility)
            is_baseline_owner = project.owner_id == current_user_id

            if row is None and not is_baseline_owner:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="You are not a member of this project"
                )

            # Extract counts (default to 1 if the members table is empty but baseline owner exists)
            target = row[0] if row else None
            owner_count = row[1] if row else (1 if is_baseline_owner else 0)
            total_count = row[2] if row else (1 if is_baseline_owner else 0)
            
            is_effective_owner = is_baseline_owner or (target and target.role == 'owner')

            if is_effective_owner and owner_count == 1 and total_count > 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="You are the only owner. Transfer ownership before leaving."
                )
                
            # If total_count is 1 and they are the owner, leaving implies deleting the whole project
            if is_effective_owner and total_count <= 1:
                # Delete project (which will cascade to members due to ForeignKey ON DELETE CASCADE)
                await self.delete(id=project_id)
            else:
                # Otherwise, just remove this member
                delete_stmt = (
                    delete(ProjectMember)
                    .where(
                        and_(
                            ProjectMember.project_id == project_id,
                            ProjectMember.user_id == current_user_id
                        )
                    )
                )
                await self.db.execute(delete_stmt)
                
            await self.db.commit()
            return True

        except HTTPException:
            raise
        except Exception as e:
            await self.db.rollback()
            raise e
    