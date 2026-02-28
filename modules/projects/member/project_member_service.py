from fastapi import status, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from models.project import Project
from modules.projects.member.project_member_repository import ProjectMemberRepository
from modules.projects.project.project_repository import ProjectRepository
from schemas.project_schema import *

class ProjectMemberService:
    """
    Service layer for ProjectMember management.
    
    Flow:
    1. Handle business logic like API key generation.
    2. Interface with ProjectMemberRepository for DB operations.
    3. Maintain clean separation between API and DB.
    """
    def __init__(self, db: AsyncSession):
        self.db = db
        self.project_repository = ProjectRepository(db)
        self.project_member_repo = ProjectMemberRepository(db)


    async def update_user_role(
        self, project_id: UUID, user_id: UUID, 
        role: str, updator_id: UUID
    ) -> bool:
        """
        Update member role (member/owner)
        Validation:
            Updator should be a project owner
            Target user should be in project_members
        """
        try:
            # Verify project exists
            project = await self.project_repository.get_by_id(id=project_id)
            if project is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Project not found: {project_id}"
                )

            # All member validation + update handled in one repo call
            success = await self.project_member_repo.patch_member_role(
                user_id=user_id,
                role=role,
                current_user_id=updator_id,
                project=project
            )

            if not success:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to update role to '{role}'"
                )

            return True

        except HTTPException:
            raise
        except Exception as e:
            await self.db.rollback()
            raise e
        
    
    async def remove_member(self, user_id: UUID, current_user_id: UUID, project_id: UUID) -> bool:
        """
        Remove member for user only role
        - verify project exists
        - verify if user to be removed is a project member
        - verify current user role
        """
        try:
            # verify project exists
            project: Project = await self.project_repository.get_by_id(id=project_id)
            if project is None:
                raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Project not found: {project_id}"
                    )
            
            # validate action
            success = await self.project_member_repo.delete_member(
                user_id=user_id, current_user_id=current_user_id, 
                project=project
            )
            
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to remove a user to the project."
                )

            return True
        
        except HTTPException:
            raise
        except Exception as e:
            await self.db.rollback()
            raise e
    