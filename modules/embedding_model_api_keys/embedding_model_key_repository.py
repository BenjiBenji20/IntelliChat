from sqlalchemy.ext.asyncio import AsyncSession

from base.base_crud_repository import BaseCrudRepository
from models.embedding_model_key import EmbeddingModelKey


class EmbeddingModelKeyRepository(BaseCrudRepository[EmbeddingModelKey]):
    """
    EmbeddingModelKey Repository for direct DB operations.
    
    Inherits standardized CRUD from BaseCrudRepository.
    """
    def __init__(self, db: AsyncSession) -> None:
        super().__init__(EmbeddingModelKey, db)
        