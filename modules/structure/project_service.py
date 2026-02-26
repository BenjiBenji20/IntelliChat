from fastapi import status, HTTPException
import secrets
import hashlib
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from configs.settings import settings
from models.project import Project
from modules.structure.project_repository import ProjectRepository
from schemas.project_schema import *

class ProjectService:
    """
    Service layer for Project management.
    
    Flow:
    1. Handle business logic like API key generation.
    2. Interface with ProjectRepository for DB operations.
    3. Maintain clean separation between API and DB.
    """
    def __init__(self, db: AsyncSession):
        self.db = db
        self.project_repository = ProjectRepository(db)
    
    async def create_project(self, owner_id: UUID, data: CreateProjectSchema) -> ResponseProjectSchema:
        """
        Create a new project with a generated API key.
        
        Flow:
        1. Generate unique API key and its SHA-256 hash.
        2. Combine user input with owner_id and hashed key.
        3. Save to database using repository.
        4. Return response with the raw API key (one-time view).
        """
        try:
            # Generate API key and hashed version
            api_key, hashed_api_key = self.generate_api_key()
            
            # Prepare project data
            project_data = data.model_dump()
            project_data.update({
                "owner_id": owner_id,
                "hashed_api_key": hashed_api_key
            })
            
            # Persist to DB
            project: Project = await self.project_repository.create(**project_data)
            
            return ResponseProjectSchema(
                id=project.id,
                owner_id=project.owner_id,
                name=project.name,
                description=project.description,
                is_active=project.is_active,
                api_key=api_key, # Return raw key only once
                created_at=project.created_at,
                updated_at=project.updated_at
            )
            
        except IntegrityError:
            await self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Project name already exists for this owner."
            )
        except Exception as e:
            await self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Internal server error: {str(e)}"
            )

    async def regenerate_api_key(self, project_id: UUID, owner_id: UUID) -> dict:
        """
        Rotate the API key for a project.
        
        Flow:
        1. Verify project exists and belongs to owner.
        2. Generate new keys.
        3. Update hashed_api_key in DB.
        4. Return new raw key.
        """
        project = await self.project_repository.get_by_id(project_id)
        if not project or project.owner_id != owner_id:
            raise HTTPException(status_code=404, detail="Project not found")
            
        api_key, hashed_api_key = self.generate_api_key()
        await self.project_repository.update(project_id, hashed_api_key=hashed_api_key)
        
        return {"api_key": api_key}
        
    # =============================================
    # HELPER METHODS
    # =============================================
    def generate_api_key(self) -> tuple[str, str]:
        """
        Generate a secure API key and its hash.
        
        Flow:
            api_key: to show to user once and use as verifier
            hashed_api_key: to store to db
            req[api_key] -> 64 char 256 encoding match hashed_api_key stored in db 
        """
        api_key = settings.SECRET_KEY_PREFIX + secrets.token_urlsafe(32)
        hashed_api_key = hashlib.sha256(api_key.encode()).hexdigest()

        return api_key, hashed_api_key
    