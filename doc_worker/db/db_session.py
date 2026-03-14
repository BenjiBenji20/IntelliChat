from typing import AsyncGenerator
import asyncpg
from doc_worker.configs.settings import settings

pool: asyncpg.Pool = None

async def init_db_pool():
    global pool
    pool = await asyncpg.create_pool(
        dsn=settings.DATABASE_URL,
        min_size=2,
        max_size=5,
    )

async def close_db_pool():
    global pool
    await pool.close()

# use for route depends for async db connection
async def get_async_db() -> AsyncGenerator[asyncpg.Connection, None]:
    if pool is None:
        raise RuntimeError("Database pool is not initialized.")
    
    async with pool.acquire() as session:
        yield session
