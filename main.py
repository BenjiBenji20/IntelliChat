from contextlib import asynccontextmanager
from fastapi import FastAPI

from configs.settings import settings
from db.base import Base
from db.db_session import engine

@asynccontextmanager
async def life_span(app: FastAPI):
    try:
        async with engine.begin() as conn:
            # await conn.run_sync(Base.metadata.drop_all)
            # await conn.run_sync(Base.metadata.create_all)

            print("\n\n Postgres connected successfully!")
            
        yield
        
    finally:
        await engine.dispose()
        print("\n\Postgres acyncpg engine disposed...")
        print("Application shutdown...")
        
        
app = FastAPI(
    title=settings.APP_NAME,
    lifespan=life_span
)
