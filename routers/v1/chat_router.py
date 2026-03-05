from uuid import UUID

from fastapi import APIRouter, Request, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from db.db_session import get_async_db
from dependencies.auth import get_current_user
from dependencies.chatbot_secret_key import intellichat_secret
from modules.chatbot.chat.chat_service import ChatService
from schemas.chatbot_schema import RequestChat, ResponseChat

router = APIRouter(
    # prefix="/api/chat" no prefix. this is user base /{domain}/{project_id}/{chatbot_id}
    tags=["Chatbot chat"]
)


@router.post("/api/chat-ai/test/{project_id}/{chatbot_id}", response_model=ResponseChat)
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
    service = ChatService(db)
    return await service.chat(
        chat=chat,
        project_id=project_id,
        chatbot_id=chatbot_id,
        environment="development"
    )


@router.post("/{project_id}/{chatbot_id}", response_model=ResponseChat)
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
    service = ChatService(db)
    return await service.chat(
        chat=chat,
        project_id=project_id,
        chatbot_id=chatbot_id,
        environment="production"
    )
