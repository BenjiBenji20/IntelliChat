from uuid import UUID

from fastapi import APIRouter, Request, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from db.db_session import get_async_db
from dependencies.auth import get_current_user
from dependencies.chatbot_secret_key import intellichat_secret
from modules.chatbot.chatbot_service import ChatbotService
from modules.chatbot.chatbot_schema import *
from dependencies.rate_limit import rate_limit_by_user, rate_limit_by_api_key

router = APIRouter(
    # prefix="/api/chat" no prefix. this is user base /{domain}/{project_id}/{chatbot_id}
    tags=["Chatbot chat"]
)

@router.get(
    "/api/chat-ai/{project_id}/chatbot-state",
    response_model=ChatbotStateSchema,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(rate_limit_by_user())]
)
async def check_chatbot_step(
    project_id: UUID,
    db: AsyncSession = Depends(get_async_db),
    current_user_id: UUID = Depends(get_current_user)
):
    service = ChatbotService(db)
    return await service.check_chatbot_step(project_id=project_id)


@router.post(
    "/api/chat-ai/create/identity", 
    response_model=ResponseChatbotSchema, 
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(rate_limit_by_user())]
)
async def create_chatbot_identity(
    payload: CreateRequestChatbotSchema,
    db: AsyncSession = Depends(get_async_db),
    current_user_id: UUID = Depends(get_current_user)
):
    payload.user_id = current_user_id
    service = ChatbotService(db)
    return await service.create_chatbot_identity(payload)


@router.patch(
    "/api/chat-ai/update/identity",
    response_model=ResponseChatbotSchema,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(rate_limit_by_user())]
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
    service = ChatbotService(db)
    return await service.update_chatbot_identity(payload)


@router.post(
    "/api/chat-ai/test/{project_id}/{chatbot_id}", 
    response_model=ResponseChat,
    dependencies=[Depends(rate_limit_by_user())]
)
async def test_intellichat(
    project_id: UUID,
    chatbot_id: UUID,
    chat: RequestChat,
    current_user_id: UUID = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    IntelliChat test your chatbot in Overview page    
    """
    service = ChatbotService(db)
    return await service.chat(
        chat=chat,
        project_id=project_id,
        chatbot_id=chatbot_id,
        environment="development"
    )


@router.post(
    "/{project_id}/{chatbot_id}", 
    response_model=ResponseChat,
    dependencies=[Depends(rate_limit_by_api_key())]
)
async def intellichat(
    project_id: UUID,
    chatbot_id: UUID,
    chat: RequestChat,
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    _: None = Depends(intellichat_secret)
):
    """
    User/production endpoint    
    """
    service = ChatbotService(db)
    return await service.chat(
        chat=chat,
        project_id=project_id,
        chatbot_id=chatbot_id,
        environment="production"
    )
