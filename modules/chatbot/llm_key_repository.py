from sqlalchemy.ext.asyncio import AsyncSession

from base.base_crud_repository import BaseCrudRepository
from models.llm_key import LlmKey


class LlmKeyRepository(BaseCrudRepository[LlmKey]):
    """
    LlmKey Repository for direct DB operations.
    
    Inherits standardized CRUD from BaseCrudRepository.
    """
    def __init__(self, db: AsyncSession) -> None:
        super().__init__(LlmKey, db)
        