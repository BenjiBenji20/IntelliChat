from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from db.db_session import get_async_db
from dependencies.auth import get_current_user
from modules.chatbot.chatbot_service import ChatbotService
from modules.embedding_model_api_keys.embedding_model_api_keys_service import EmbeddingModelAPIKeyService
from modules.llm_api_keys.llm_api_keys_service import ChatbotAPIKeyService
from modules.llm_api_keys.llm_api_keys_schema import *
from modules.chatbot.chatbot_schema import *
from modules.embedding_model_api_keys.embedding_model_api_keys_schema import *


router = APIRouter(
    prefix="/api/chatbot/api-key",
    tags=["Chatbot API Keys"]
)

@router.post(
    "/upload-llm-key", 
    response_model=ResponseLlmSchema, 
    status_code=status.HTTP_201_CREATED
)
async def upload_llm_key(
    payload: CreateRequestLlmSchema,
    db: AsyncSession = Depends(get_async_db),
    current_user_id: UUID = Depends(get_current_user)
):
    payload.user_id = current_user_id
    service = ChatbotAPIKeyService(db)
    return await service.upload_llm_key(payload)


@router.post(
    "/upload-embedding-model-key", 
    response_model=ResponseEmbbedingModelSchema, 
    status_code=status.HTTP_201_CREATED
)
async def upload_embedding_model_key(
    payload: CreateRequestEmbbedingModelSchema,
    db: AsyncSession = Depends(get_async_db),
    current_user_id: UUID = Depends(get_current_user)
):
    payload.user_id = current_user_id
    service = EmbeddingModelAPIKeyService(db)
    return await service.upload_embedding_model_key(payload)


@router.patch(
    "/update/llm-key",
    response_model=ResponseLlmSchema,
    status_code=status.HTTP_200_OK
)
async def update_llm_api_key(
    payload: UpdateRequestLlmSchema,
    db: AsyncSession = Depends(get_async_db),
    current_user_id: UUID = Depends(get_current_user)
):
    """
    Patch llm_keys
    original id, chatbot_id, created_at and user_id will persist
    """
    service = ChatbotAPIKeyService(db)
    return await service.update_llm_api_key(payload)


@router.patch(
    "/update/embedding-model-key",
    response_model=ResponseEmbbedingModelSchema,
    status_code=status.HTTP_200_OK
)
async def update_embedding_model_api_key(
    payload: UpdateRequestEmbeddingModelSchema,
    db: AsyncSession = Depends(get_async_db),
    current_user_id: UUID = Depends(get_current_user)
):
    """
    Patch embedding_model_keys
    original id, chatbot_id, created_at and user_id will persist
    """
    service = EmbeddingModelAPIKeyService(db)
    return await service.update_embedding_model_api_key(payload)
