from base.base_crud_repository import BaseCrudRepository
from models.project import Project

from sqlalchemy.ext.asyncio import AsyncSession


class ProjectRepository(BaseCrudRepository[Project]):
    """
    Project Repository for direct DB operations.
    
    Inherits standardized CRUD from BaseCrudRepository.
    """
    def __init__(self, db: AsyncSession) -> None:
        super().__init__(Project, db)
