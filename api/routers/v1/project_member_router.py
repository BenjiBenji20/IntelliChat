from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from api.db.db_session import get_async_db
from api.dependencies.auth import get_current_user
from api.modules.projects.member.project_member_service import ProjectMemberService
from api.dependencies.rate_limit import rate_limit_by_user

router = APIRouter(
    prefix="/api/project-members",
    tags=["Project Members"]
)

@router.patch(
    "/update-role/{project_id}/{user_id}/{new_role}", 
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(rate_limit_by_user())]
)
async def update_user_role(
    project_id: UUID,
    user_id: UUID, # ProjectMember.user_id
    new_role: str,
    db: AsyncSession = Depends(get_async_db),
    current_user_id: UUID = Depends(get_current_user)
):
    """
    Update member role (member/owner)
    """
    service = ProjectMemberService(db)
    return await service.update_user_role(
            project_id=project_id, user_id=user_id, 
            role=new_role, updator_id=current_user_id
        )


@router.delete(
    "/delete/{user_id}/{project_id}", 
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(rate_limit_by_user())]
)
async def remove_member(    
    user_id: UUID, # ProjectMember.user_id
    project_id: UUID,
    db: AsyncSession = Depends(get_async_db),
    current_user_id: UUID = Depends(get_current_user)
):
    """
    Remove member (owner only role)
    Leave from group (allowed for self leave)
    """
    service = ProjectMemberService(db)
    return await service.remove_member(
        user_id=user_id, current_user_id=current_user_id, 
        project_id=project_id
    )
 