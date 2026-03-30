import typing
from uuid import UUID

from fastapi import APIRouter, Request, Depends, status
from fastapi.responses import StreamingResponse
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncSession

from api.configs.qdrant import get_qdrant_client
from api.db.db_session import get_async_db
from api.dependencies.auth import get_current_user
from api.dependencies.chatbot_secret_key import intellichat_secret
from api.modules.chat.chat_schema import IntelliChatRequest, IntellichatResponseSchema
from api.modules.chat.test_intellichat_service import TestIntelliChatService
from api.modules.chatbot.chatbot_service import ChatbotService
from api.modules.chat.intellichat_service import IntelliChatService
from api.modules.chatbot.chatbot_schema import *
from api.dependencies.rate_limit import rate_limit_by_user, rate_limit_by_api_key

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
async def chatbot_current_state(
    project_id: UUID,
    db: AsyncSession = Depends(get_async_db),
    current_user_id: UUID = Depends(get_current_user)
):
    service = ChatbotService(db)
    return await service.chatbot_current_state(project_id=project_id)


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
    response_model=IntellichatResponseSchema,
    dependencies=[Depends(rate_limit_by_user())]
)
async def test_intellichat(
    project_id: UUID,
    chatbot_id: UUID,
    payload: IntelliChatRequest,
    db: AsyncSession = Depends(get_async_db),
    qdrant: AsyncQdrantClient = Depends(get_qdrant_client),
    _: UUID = Depends(get_current_user)
):
    """
    IntelliChat test your chatbot in Overview page    
    """
    service = TestIntelliChatService(db=db, qdrant=qdrant)
    res = await service.test_chat(
        query=payload.query,
        top_k=payload.top_k,
        conversation_id=payload.conversation_id,
        filters=payload.filters if payload.filters else None,
        project_id=project_id,
        chatbot_id=chatbot_id,
    )
    if isinstance(res, typing.AsyncGenerator):
        return StreamingResponse(res, media_type="text/event-stream")
    return res


@router.post(
    "/{project_id}/{chatbot_id}", 
    response_model=IntellichatResponseSchema,
    dependencies=[Depends(rate_limit_by_api_key())]
)
async def intellichat(
    project_id: UUID,
    chatbot_id: UUID,
    payload: IntelliChatRequest,
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    qdrant: AsyncQdrantClient = Depends(get_qdrant_client),
    _: None = Depends(intellichat_secret)
):
    """
    User/production endpoint    
    """
    service = IntelliChatService(db=db, qdrant=qdrant)
    res = await service.chat(
        query=payload.query,
        stream=payload.stream,
        top_k=payload.top_k,
        conversation_id=payload.conversation_id,
        filters=payload.filters if payload.filters else None,
        project_id=project_id,
        chatbot_id=chatbot_id,
    )
    if isinstance(res, typing.AsyncGenerator):
        return StreamingResponse(res, media_type="text/event-stream")
    return res
