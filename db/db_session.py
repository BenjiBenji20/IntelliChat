from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from configs.settings import settings

engine = create_async_engine(
    settings.DATABASE_URL, 
    echo=settings.DEBUG,
    pool_size=5,              
    max_overflow=10,
    pool_pre_ping=True,       # Verify connections before using
    pool_recycle=3600,        # Recycle connections every hour
)
async_session = async_sessionmaker(engine, expire_on_commit=False)

# use for route depends for async db connection/session
async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
