from base.base_crud_repository import BaseCrudRepository
from models.project_member import ProjectMember

from sqlalchemy.ext.asyncio import AsyncSession

from models.project_member import ProjectMember
from modules.structure.project_repository import ProjectRepository


class ProjectMemberRepository(BaseCrudRepository[ProjectMember]):
    """
    ProjectMember Repository for direct DB operations.
    
    Inherits standardized CRUD from BaseCrudRepository.
    """
    def __init__(self, db: AsyncSession) -> None:
        super().__init__(ProjectMember, db)
        self.project_repo = ProjectRepository(db)
        