from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from db.db_session import get_async_db
from dependencies.auth import get_current_user
from modules.projects.project.project_service import ProjectService
from schemas.project_schema import (
    CreateProjectSchema,
    ResponseProjectSchema,
)

router = APIRouter(
    prefix="/api/projects",
    tags=["Projects"]
)

@router.post("/create", response_model=ResponseProjectSchema, status_code=status.HTTP_201_CREATED)
async def create_project(
    payload: CreateProjectSchema,
    db: AsyncSession = Depends(get_async_db),
    user_id: UUID = Depends(get_current_user) # auth().id
):
    """Create a new project."""
    service = ProjectService(db)
    return await service.create_project(owner_id=user_id, data=payload)

    
@router.post("/{project_id}/api-key/{user_id}/new", status_code=status.HTTP_201_CREATED)
async def regenerate_api_key(
    project_id: UUID,
    user_id: UUID, # ProjectMember.user_id
    db: AsyncSession = Depends(get_async_db),
    current_user_id: UUID = Depends(get_current_user)
):
    """
    Regenerate new api key of the project
    """
    service = ProjectService(db)
    return await service.regenerate_api_key(
        project_id=project_id,
        user_id=user_id,
        current_user_id=current_user_id
    )
    
    
@router.post("/leave/{project_id}", status_code=status.HTTP_202_ACCEPTED)
async def leave_from_project(  
    project_id: UUID,
    db: AsyncSession = Depends(get_async_db),
    current_user_id: UUID = Depends(get_current_user) # auth().id
):
    """
    Leave from project(any role)
    """
    service = ProjectService(db)
    return await service.leave_from_project(
        project_id=project_id, user_id=current_user_id
    )
    