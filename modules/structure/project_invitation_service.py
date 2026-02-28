from fastapi import status, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from models.profile import Profile
from modules.structure.project_invitation_repository import ProjectInvitationRepository
from modules.structure.project_repository import ProjectRepository
from modules.structure.project_member_repository import ProjectMemberRepository
from schemas.project_invitation_schema import *
from schemas.project_schema import *

class ProjectInvitationService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.project_invitation_repo = ProjectInvitationRepository(db)
        self.project_repo = ProjectRepository(db)
        self.project_member_repo = ProjectMemberRepository(db)
        
    async def create_project_invitations(
        self, invitation_data: CreateProjectInvitationSchema
    ) -> ResponseCreateProjectInvitationSchema:
        """
        Bulk user invitation
        
        Validations:
            - project 
            - cannot invite if project deleted
            - owner cannot self invite
            - cannot invite existing 
            - username exists
            - caller must be a project owner
            - project owner only role
            - cannot invite already pending invitation
        """
        try:
            data_dict = invitation_data.model_dump()
            
            invitation_response: dict = await self.project_invitation_repo.create_project_invitations(
                **data_dict
            )
            
            if invitation_response is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Project invitation request fail."
                )
                
            return ResponseCreateProjectInvitationSchema(
                invited_users=invitation_response.get('invited_users', []),
                failed_invitations=invitation_response.get('failed_invitations', [])
            )
            
        except IntegrityError:
            await self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User cannot go through to the project."
            )
        except Exception as e:
            await self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Internal server error: {str(e)}"
            )
            
    
    async def update_user_role(self, id: UUID, role: str, updator_id: UUID) -> bool:
        """
        Update member role (member/owner)
        Validation:
            Updator should be a project owner
            User to be update should be in project_members
        """
        try:
            project_member: Profile = await self.project_repo.get_project_member_by_id(id)
            if project_member is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Project member not found: {id}"
                )
                
            updator: Profile = await self.project_repo.get_project_member_by_id(updator_id)
            if updator is None or updator.role != 'owner':
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Updator role {updator.role} not allowed to update: {id}"
                )
                
            updated_user = await self.project_member_repo.update(role, id=id)
            
            if updated_user is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Project member role {role} update failed."
                )
                
            return True
    
        except HTTPException:
            raise
        except Exception as e:
            await self.db.rollback()
            raise e
        