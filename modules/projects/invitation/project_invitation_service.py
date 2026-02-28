from fastapi import status, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from modules.projects.invitation.project_invitation_repository import ProjectInvitationRepository
from modules.projects.project.project_repository import ProjectRepository
from schemas.project_invitation_schema import *
from schemas.project_schema import *

class ProjectInvitationService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.project_invitation_repo = ProjectInvitationRepository(db)
        self.project_repo = ProjectRepository(db)
        
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
    