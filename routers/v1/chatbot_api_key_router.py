from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from db.db_session import get_async_db
from dependencies.auth import get_current_user
from modules.chatbot.create.chatbot_api_key_service import ChatbotAPIKeyService
from schemas.chatbot_schema import *

router = APIRouter(
    prefix="/api/chatbot/api-key",
    tags=["Chatbot API Keys"]
)

@router.get(
    "/{project_id}/chatbot-state",
    response_model=ChatbotStateSchema,
    status_code=status.HTTP_200_OK
)
async def check_chatbot_step(
    project_id: UUID,
    db: AsyncSession = Depends(get_async_db),
    current_user_id: UUID = Depends(get_current_user)
):
    service = ChatbotAPIKeyService(db)
    return await service.check_chatbot_step(project_id=project_id)


@router.post(
    "/create-identity", 
    response_model=ResponseChatbotSchema, 
    status_code=status.HTTP_201_CREATED
)
async def create_chatbot_identity(
    payload: CreateRequestChatbotSchema,
    db: AsyncSession = Depends(get_async_db),
    current_user_id: UUID = Depends(get_current_user)
):
    payload.user_id = current_user_id
    service = ChatbotAPIKeyService(db)
    return await service.create_chatbot_identity(payload)


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
    service = ChatbotAPIKeyService(db)
    return await service.upload_embedding_model_key(payload)


@router.patch(
    "/update/identity",
    response_model=ResponseChatbotSchema,
    status_code=status.HTTP_200_OK
)
async def update_chatbot_identity(
    payload: UpdateRequestChatbotSchema,
    db: AsyncSession = Depends(get_async_db),
    current_user_id: UUID = Depends(get_current_user)
):
    """
    Patch chatbots
    original id, user_id, created_at and project_id will persist
    """
    service = ChatbotAPIKeyService(db)
    return await service.update_chatbot_identity(payload)


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
    service = ChatbotAPIKeyService(db)
    return await service.update_embedding_model_api_key(payload)
