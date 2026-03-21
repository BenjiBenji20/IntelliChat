import hashlib
from uuid import UUID
import asyncio

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import and_, select

from api.configs.settings import settings
from api.db.db_session import get_async_db
from api.models.project import Project
from api.modules.cache.redis_service import redis_service

secret_key_scheme = APIKeyHeader(name=settings.API_KEY_HEADER_NAME)

async def intellichat_secret(
        project_id: UUID,
        secret_key: str = Depends(secret_key_scheme),
        db: AsyncSession = Depends(get_async_db)
    ) -> None:
    """
    Verify secret key from header against encrypted key in db
    request key [raw] -> hashed raw key -> db [encrypted key] = 
    match [hashed request key == decoded key, project_id == id ??]
    """

    """
    TODO:
        cache secret key ttl=1hr
    """
    try:
        if not settings.ENCRYPTION_KEY:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="System encryption key not found."
            )
    
        # Hash the incoming raw key
        hashed_key = hashlib.sha256(secret_key.encode()).hexdigest()
        
        # get hashed_key. If hit, return True
        cached_secret = await redis_service.get(key=str(project_id), prefix="secret_key")
        if cached_secret:
            return True

        # query in db using project_id
        stmt = (
            select(Project.hashed_api_key)
            .where(
                and_(
                    Project.id == project_id,
                    Project.hashed_api_key == hashed_key,
                    Project.is_active == True
                )
            )
        )

        result = await db.execute(stmt)
        
        is_secret_valid = True if result.scalar_one_or_none() is not None else False

        if not is_secret_valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API Key. Contact the project owners to solve this issue."
            )
            
        # cache hashed_key in redis. non-blocking coroutine
        asyncio.create_task(
            redis_service.set(
                value=hashed_key, ttl=3600,
                key=str(project_id), prefix="secret_key"
            )
        )
        
        return is_secret_valid

    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise e
    