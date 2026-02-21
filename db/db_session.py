from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from configs.settings import settings

engine = create_async_engine(
        settings.DATABASE_URL, 
        echo=settings.DEBUG,
    ) # for async app
async_session = async_sessionmaker(engine, expire_on_commit=False)

# use for route depends for async db connection/session
async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        yield session
