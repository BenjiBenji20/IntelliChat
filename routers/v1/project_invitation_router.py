from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from db.db_session import get_async_db
from dependencies.auth import get_current_user
from modules.projects.invitation.project_invitation_service import ProjectInvitationService
from modules.projects.invitation.project_invitation_schema import *

router = APIRouter(
    prefix="/api/project-invitations",
    tags=["Project Invitations"]
)

@router.post(
    "/create-many", 
    response_model=ResponseCreateProjectInvitationSchema, 
    status_code=status.HTTP_201_CREATED
)
async def create_project_invitations(
    payload: CreateProjectInvitationSchema,
    db: AsyncSession = Depends(get_async_db),
    current_user_id: UUID = Depends(get_current_user)
):
    """
    Bulk user invitation
    """
    service = ProjectInvitationService(db)
    return await service.create_project_invitations(payload)
