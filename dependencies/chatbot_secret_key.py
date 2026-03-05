import hashlib
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import and_, select

from configs.settings import settings
from db.db_session import get_async_db
from models.project import Project

secret_key_scheme = APIKeyHeader(name="Intellichat-Secret")

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
    try:
        if not settings.ENCRYPTION_KEY:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="System encryption key not found."
            )
    
        # Hash the incoming raw key
        hashed_key = hashlib.sha256(secret_key.strip(settings.SECRET_KEY_PREFIX).encode()).hexdigest()

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

    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise e
    