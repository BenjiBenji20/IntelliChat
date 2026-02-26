from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from db.db_session import get_async_db
from dependencies.auth import get_current_user
from modules.structure.project_service import ProjectService
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
    user_id: UUID = Depends(get_current_user)
):
    """Create a new project."""
    service = ProjectService(db)
    return await service.create_project(owner_id=user_id, data=payload)


@router.patch("/regenerate-key/{id}")
async def regenerate_api_key(
    id: UUID,
    db: AsyncSession = Depends(get_async_db),
    user_id: UUID = Depends(get_current_user)
):
    """Regenerate the API key for a project."""
    service = ProjectService(db)
    return await service.regenerate_api_key(project_id=id, owner_id=user_id)
