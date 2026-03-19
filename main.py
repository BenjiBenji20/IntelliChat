import os
import logging

# cloud deployment logging 
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug(f"Starting app - PORT from env: {os.environ.get('PORT', 'not set')}")
logger.debug(f"Binding to 0.0.0.0:{os.environ.get('PORT', '8080')}")

from contextlib import asynccontextmanager
from fastapi import FastAPI
from sqlalchemy import text
from fastapi.middleware.cors import CORSMiddleware

from api.configs.settings import settings
from api.db.db_session import engine

# Import all models 
from api.db.base import Base
from api.models.profile import Profile
from api.models.chatbot import Chatbot
from api.models.document import Document
from api.models.embedding_metadata import EmbeddingMetadata
from api.models.conversation import Conversation
from api.models.message import Message
from api.models.api_key import ApiKey
from api.models.redis_key import RedisKey
from api.models.embedding_model_key import EmbeddingModelKey
from api.models.llm_key import LlmKey
from api.models.project import Project
from api.models.project_member import ProjectMember
from api.models.project_invitation import ProjectInvitation
from api.models.chatbot_behavior import ChatbotBehavior

from api.configs.qdrant import init_qdrant_client

@asynccontextmanager
async def life_span(app: FastAPI):
    try:
        async with engine.begin() as conn:
            # await conn.run_sync(Base.metadata.drop_all)
            # await conn.run_sync(Base.metadata.create_all)
            await conn.execute(text("SELECT 1"))
            print("\n\nPostgres connected successfully!")
            
            await init_qdrant_client()
            print("\n\nQDrant connected successfully!")
            
        yield
        
    finally:
        await engine.dispose()
        print("\n\nPostgres acyncpg engine disposed...")
        print("Application shutdown...")
        
        
app = FastAPI(
    title=settings.APP_NAME,
    lifespan=life_span
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.CORS_DEV_ORIGIN, settings.CORS_PROD_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],  # Allows all headers
)

 
# Register routers
from api.routers.v1.project_router import router as project_router
from api.routers.v1.project_invitation_router import router as project_invitation_router
from api.routers.v1.project_member_router import router as project_member_router
from api.routers.v1.chatbot_api_key_router import router as chatbot_api_key_router
from api.routers.v1.chatbot_router import router as chat_ai_router
from api.routers.v1.behavior_studio_router import router as behavior_studio_router
from api.routers.v1.document_router import router as document_router
from api.routers.v1.retrieval_router import router as retrieval_router
app.include_router(project_router)
app.include_router(project_invitation_router)
app.include_router(project_member_router)
app.include_router(chatbot_api_key_router)
app.include_router(chat_ai_router)
app.include_router(behavior_studio_router)
app.include_router(document_router)
app.include_router(retrieval_router)
