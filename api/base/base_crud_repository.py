from typing import Generic, TypeVar, Type, List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func

from api.db.base import Base

ModelType = TypeVar("ModelType", bound=Base)

class BaseCrudRepository(Generic[ModelType]):
    """
    Async base repository with common CRUD operations.
    
    Flow:
    1. Initialize with specific SQLAlchemy model and session.
    2. Provide standardized methods for Create, Read, Update, Delete.
    3. Handle common database errors gracefully.
    """
    
    def __init__(self, model: Type[ModelType], db: AsyncSession):
        self.model = model
        self.db = db
    
    # ============================================
    # CREATE
    # ============================================
    async def create(self, **kwargs) -> ModelType:
        """
        Create a new record.
        
        Steps:
        1. Normalize data (e.g., lowercasing email).
        2. Instantiate model with provided keywords.
        3. Add to session and commit.
        4. Refresh to get DB-generated fields like ID.
        """
        try:
            if "email" in kwargs and kwargs["email"]:
                kwargs["email"] = kwargs["email"].lower().strip()
            
            instance = self.model(**kwargs)
            self.db.add(instance)
            await self.db.commit()
            await self.db.refresh(instance)
            return instance
        except Exception as e:
            await self.db.rollback()
            raise e
    
    
    async def create_many(self, objects: List[Dict[str, Any]]) -> List[ModelType]:
        """
        Create multiple records in a single transaction.
        
        Steps:
        1. Batch instantiate models.
        2. Add all to session and commit.
        3. Refresh all instances.
        """
        try:
            instances = [self.model(**obj) for obj in objects]
            self.db.add_all(instances)
            await self.db.commit()
            for instance in instances:
                await self.db.refresh(instance)
            return instances
        except Exception as e:
            await self.db.rollback()
            raise e
    
    
    # ============================================
    # READ
    # ============================================
    async def get_by_id(self, id: Any) -> Optional[ModelType]:
        """
        Fetch a single record by its primary key.
        
        Steps:
        1. Execute select query with ID filter.
        2. Return the first result or None.
        """
        result = await self.db.execute(
            select(self.model).where(self.model.id == id)
        )
        return result.scalars().first()
    
    
    async def get_by_field(self, field: str, value: Any) -> Optional[ModelType]:
        """Get single record by any field."""
        result = await self.db.execute(
            select(self.model).where(getattr(self.model, field) == value)
        )
        return result.scalars().first()
    
    
    async def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        order_by: Optional[str] = None,
        desc: bool = False
    ) -> List[ModelType]:
        """Get all records with pagination."""
        query = select(self.model)
        
        if order_by:
            order_field = getattr(self.model, order_by)
            if desc:
                order_field = order_field.desc()
            query = query.order_by(order_field)
        
        query = query.offset(skip).limit(limit)
        result = await self.db.execute(query)
        return result.scalars().all()
    
    
    async def filter_by(self, **filters) -> List[ModelType]:
        """Filter records by multiple fields."""
        query = select(self.model).filter_by(**filters)
        result = await self.db.execute(query)
        return result.scalars().all()
    
    
    async def count(self, **filters) -> int:
        """Count records matching filters."""
        query = select(func.count()).select_from(self.model)
        if filters:
            for key, value in filters.items():
                query = query.where(getattr(self.model, key) == value)
        result = await self.db.execute(query)
        return result.scalar()
    
    
    async def exists(self, **filters) -> bool:
        """Check if record exists."""
        query = select(self.model).filter_by(**filters).limit(1)
        result = await self.db.execute(query)
        return result.scalars().first() is not None
    
    
    # ============================================
    # UPDATE
    # ============================================
    async def update(self, id: Any, **kwargs) -> Optional[ModelType]:
        """
        Update a record by ID.
        
        Steps:
        1. Find instance by ID.
        2. Iterate over keys and set attributes if they exist on model.
        3. Commit changes and refresh instance.
        """
        try:
            instance = await self.get_by_id(id)
            if not instance:
                return None
            
            for key, value in kwargs.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)
            
            await self.db.commit()
            await self.db.refresh(instance)
            return instance
        except Exception as e:
            await self.db.rollback()
            raise e
    
    
    async def update_many(self, filters: Dict[str, Any], updates: Dict[str, Any]) -> int:
        """
        Update multiple records matching filters.
        
        Steps:
        1. Build bulk update statement.
        2. Execute and commit.
        3. Return count of affected rows.
        """
        try:
            stmt = update(self.model).where(
                *[getattr(self.model, k) == v for k, v in filters.items()]
            ).values(**updates)
            
            result = await self.db.execute(stmt)
            await self.db.commit()
            return result.rowcount
        except Exception as e:
            await self.db.rollback()
            raise e
    
    # ============================================
    # DELETE
    # ============================================
    async def delete(self, id: Any) -> bool:
        """
        Delete a record by ID.
        
        Steps:
        1. Find instance.
        2. Delete from session and commit.
        """
        try:
            instance = await self.get_by_id(id)
            if not instance:
                return False
            
            await self.db.delete(instance)
            await self.db.commit()
            return True
        except Exception as e:
            await self.db.rollback()
            raise e
    
    
    async def delete_many(self, **filters) -> int:
        """Delete multiple records."""
        stmt = delete(self.model).where(
            *[getattr(self.model, k) == v for k, v in filters.items()]
        )
        
        result = await self.db.execute(stmt)
        await self.db.commit()
        return result.rowcount
    
    
    async def soft_delete(self, id: str) -> Optional[ModelType]:
        """Soft delete (set is_active=False)."""
        if not hasattr(self.model, 'is_active'):
            raise AttributeError(f"{self.model.__name__} doesn't have is_active field")
        
        return await self.update(id, is_active=False)
    