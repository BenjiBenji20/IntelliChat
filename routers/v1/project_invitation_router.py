from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from db.db_session import get_async_db
from dependencies.auth import get_current_user
from modules.structure.project_invitation_service import ProjectInvitationService
from schemas.project_invitation_schema import *

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

@router.patch("/update-role/{user_id}/{new_role}", status_code=status.HTTP_200_OK)
async def update_user_role(
    user_id: UUID,
    new_role: str,
    db: AsyncSession = Depends(get_async_db),
    current_user_id: UUID = Depends(get_current_user)
):
    """
    Update member role (member/owner)
    """
    service = ProjectInvitationService(db)
    return await service.update_user_role(
            id=user_id, role=new_role, invitee_id=current_user_id
        )

"""
TODO:
    Remove member (owner)
    Update member role (owner)
    Remove pending invitation
"""