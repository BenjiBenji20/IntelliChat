import os
import logging

# cloud deployment logging 
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug(f"Starting app - PORT from env: {os.environ.get('PORT', 'not set')}")
logger.debug(f"Binding to 0.0.0.0:{os.environ.get('PORT', '8080')}")

from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

from doc_worker.configs.settings import settings
from doc_worker.db import db_session
from doc_worker.db.db_session import init_db_pool, close_db_pool

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await init_db_pool()
        await db_session.pool.fetchrow("SELECT 1") # try fetch to trigger cold start
        print("\n\nPostgres connected successfully!")
        
        yield 
        
    finally:
        await close_db_pool()
        print("\n\nPostgres acyncpg engine disposed...")
        print("Application shutdown...")


app = FastAPI(
    title=settings.APP_NAME,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.CORS_DEV_ORIGIN, settings.CORS_PROD_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],  # Allows all headers
)
