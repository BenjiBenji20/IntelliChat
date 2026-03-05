from contextlib import asynccontextmanager
from fastapi import FastAPI
from sqlalchemy import text
from fastapi.middleware.cors import CORSMiddleware

from configs.settings import settings
from db.db_session import engine

# Import all models 
from db.base import Base
from models.profile import Profile
from models.chatbot import Chatbot
from models.document import Document
from models.embedding_metadata import EmbeddingMetadata
from models.conversation import Conversation
from models.message import Message
from models.api_key import ApiKey
from models.redis_key import RedisKey
from models.embedding_model_key import EmbeddingModelKey
from models.llm_key import LlmKey
from models.project import Project
from models.project_member import ProjectMember
from models.project_invitation import ProjectInvitation

@asynccontextmanager
async def life_span(app: FastAPI):
    try:
        async with engine.begin() as conn:
            # await conn.run_sync(Base.metadata.drop_all)
            # await conn.run_sync(Base.metadata.create_all)
            await conn.execute(text("SELECT 1"))
            print("\n\nPostgres connected successfully!")
            
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
from routers.v1.project_router import router as project_router
from routers.v1.project_invitation_router import router as project_invitation_router
from routers.v1.project_member_router import router as project_member_router
from routers.v1.chatbot_api_key_router import router as chatbot_api_key_router
from routers.v1.chat_router import router as chat_ai_router
app.include_router(project_router)
app.include_router(project_invitation_router)
app.include_router(project_member_router)
app.include_router(chatbot_api_key_router)
app.include_router(chat_ai_router)
