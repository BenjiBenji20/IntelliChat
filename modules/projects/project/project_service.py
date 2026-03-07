from fastapi import status, HTTPException
import secrets
import hashlib
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from configs.settings import settings
from models.project import Project
from models.project_member import ProjectMember
from modules.projects.member.project_member_repository import ProjectMemberRepository
from modules.projects.project.project_repository import ProjectRepository
from modules.projects.project.project_schema import *

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
        self.project_member_repo = ProjectMemberRepository(db)
    
    
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
            
            # Add creator as the initial owner in project_members
            await self.project_member_repo.create(
                project_id=project.id,
                user_id=owner_id,
                role='owner',
                invited_by=owner_id
            )
            
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


    async def regenerate_api_key(
        self, project_id: UUID,
        user_id: UUID,
        current_user_id: UUID
    ) -> dict:
        """
        Rotate the API key for a project.
        
        Flow:
        1. Validate path user_id against current_user_id.
        2. Verify project exists + fetch current user role using a single query.
        3. Check ownership permissions (fallback to project.owner_id to be safe).
        4. Regenerate and save.
        """
        try:
            if user_id != current_user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid user"
                )

            # Single query returns (Project, role_or_None)
            row = await self.project_repository.get_project_with_member(
                project_id=project_id,
                user_id=current_user_id
            )

            if row is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Project not found"
                )

            project, member_role = row

            # Determine ownership (protects against projects created before the member insert fix)
            is_owner = (project.owner_id == current_user_id) or (member_role == 'owner')

            if not is_owner:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Only owners can regenerate the API key"
                )

            api_key, hashed_api_key = self.generate_api_key()
            await self.project_repository.update(project_id, hashed_api_key=hashed_api_key)

            return {"api_key": api_key}

        except HTTPException:
            raise
        except Exception as e:
            await self.db.rollback()
            raise e
    
    
    async def leave_from_project(
        self, project_id: UUID, user_id: UUID
    ) -> bool:
        """
        Leave from project (any role)
        Actions are for current user only
        """
        try:
            success = await self.project_repository.leave_project(
                project_id=project_id, current_user_id=user_id
            )
            
            return success
            
        except HTTPException:
            raise
        except Exception as e:
            await self.db.rollback()
            raise e
        
    
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
    