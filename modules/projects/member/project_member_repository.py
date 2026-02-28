from typing import Optional
from uuid import UUID
from fastapi import HTTPException, status
from sqlalchemy import and_, delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from models.project_member import ProjectMember
from base.base_crud_repository import BaseCrudRepository
from models.project import Project
from modules.projects.project.project_repository import ProjectRepository


class ProjectMemberRepository(BaseCrudRepository[ProjectMember]):
    """
    ProjectMember Repository for direct DB operations.
    
    Inherits standardized CRUD from BaseCrudRepository.
    """
    def __init__(self, db: AsyncSession) -> None:
        super().__init__(ProjectMember, db)
        self.project_repo = ProjectRepository(db)
        
    
    async def patch_member_role(
        self, user_id: UUID, role: str,
        current_user_id: UUID, project: Project
    ) -> bool:
        try:
            # Fetch both the updator and target member in one query
            stmt = (
                select(ProjectMember)
                .where(
                    and_(
                        ProjectMember.project_id == project.id,
                        ProjectMember.user_id.in_([user_id, current_user_id])
                    )
                )
            )

            results = await self.db.execute(stmt)
            members = results.scalars().all()

            # Separate updator and target
            updator = next((m for m in members if m.user_id == current_user_id), None)
            target = next((m for m in members if m.user_id == user_id), None)

            is_users_valid = self.validate_user(
                target=target, user_id=user_id, current_user_id=current_user_id,
                actor=updator, project=project, error_detail='update'
            )
            
            if not is_users_valid:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Update user fail."
                )

            # Perform update
            update_stmt = (
                update(ProjectMember)
                .where(
                    and_(
                        ProjectMember.project_id == project.id,
                        ProjectMember.user_id == user_id
                    )
                )
                .values(role=role)
                .returning(ProjectMember)
            )

            result = await self.db.execute(update_stmt)
            await self.db.commit()
            return result.scalar_one_or_none() is not None

        except HTTPException:
            raise
        except Exception as e:
            await self.db.rollback()
            raise e
    
    
    async def delete_member(self, user_id: UUID, current_user_id: UUID, project: Project) -> bool:
        """
        Remove member for user only role
        Leave from project (allowed self removal)
        - verify project exists
        - verify if user to be removed is a project member
        - verify current user role
        """
        try:
            stmt = (
                select(ProjectMember)
                .where(
                    and_(
                        ProjectMember.project_id == project.id,
                        ProjectMember.user_id.in_([user_id, current_user_id])
                    )
                )
            )
            
            results = await self.db.execute(stmt)
            members = results.scalars().all()

            # Separate deleter and target
            deleter = next((m for m in members if m.user_id == current_user_id), None)
            target = next((m for m in members if m.user_id == user_id), None)
            
            is_users_valid = self.validate_user(
                target=target, user_id=user_id, current_user_id=current_user_id,
                actor=deleter, project=project, error_detail='delete'
            )
            
            if not is_users_valid:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Remove user fail."
                )
                
            # Perform delete stmt
            delete_stmt = (
                delete(ProjectMember)
                .where(
                    and_(
                        ProjectMember.project_id == project.id,
                        ProjectMember.user_id == user_id
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
        
    
    # ======================================================================================== 
    # HELPER METHODS
    # ======================================================================================== 
    async def get_one_record(self, project_id: UUID, user_id: UUID) -> Optional[ProjectMember]:
        try:
            stmt = (
                select(ProjectMember)
                .where(
                    and_(
                        ProjectMember.project_id == project_id,
                        ProjectMember.user_id == user_id
                    )
                )
            )
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            raise e
     
    
    def validate_user(
        self, target: ProjectMember,
        user_id: UUID, current_user_id: UUID,
        actor: ProjectMember, project: Project, error_detail: str
    ) -> bool:
        if target is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project member not found: {user_id}"
            )

        if user_id == current_user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"You cannot {error_detail} {'your own role' if error_detail == 'update' else 'yourself.'}"
            )

        # For actions on others, must be owner
        # Fallback to project.owner_id if actor isn't found in members table (backward compat)
        is_owner = (project.owner_id == current_user_id) or (actor is not None and actor.role == 'owner')
        if not is_owner:
            raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Only project owners can {error_detail} a member"
                )

        return True
    