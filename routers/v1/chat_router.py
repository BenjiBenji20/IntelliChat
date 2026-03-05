from uuid import UUID

from fastapi import APIRouter, Request, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from db.db_session import get_async_db
from dependencies.chatbot_secret_key import intellichat_secret
from modules.chatbot.chat.chat_service import ChatService
from schemas.chatbot_schema import RequestChat, ResponseChat

router = APIRouter(
    # prefix="/api/chat" no prefix. this is user base /{domain}/{project_id}/{chatbot_id}
    tags=["Chatbot chat"]
)


@router.post("/{project_id}/{chatbot_id}", response_model=ResponseChat)
async def chat(
    project_id: UUID,
    chatbot_id: UUID,
    chat: RequestChat,
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    _: None = Depends(intellichat_secret)
):
    service = ChatService(db)
    return await service.chat(
        chat=chat,
        project_id=project_id,
        chatbot_id=chatbot_id
    )

"""
TODO:
    - test endpoint
"""
