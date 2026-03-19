from uuid import UUID
 
from fastapi import APIRouter, Depends, status
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncSession
 
from api.configs.qdrant import get_qdrant_client
from api.db.db_session import get_async_db
from api.dependencies.auth import get_current_user
from api.dependencies.rate_limit import rate_limit_by_user
from api.modules.retrievals.retrieval_schema import (
    RetrievalRequestSchema,
    RetrievalResponseSchema,
)
from api.modules.retrievals.retrieval_service import RetrieveEmbeddingsService
 
router = APIRouter(
    prefix="/api/retrieval",
    tags=["Retrieval"],
)
 
 
@router.post(
    "/test/{chatbot_id}",
    status_code=status.HTTP_200_OK,
    summary="Test embedding query and retrieve top-k chunks from Qdrant",
    dependencies=[Depends(rate_limit_by_user())]
)
async def retrieve_embeddings(
    chatbot_id: UUID,
    payload: RetrievalRequestSchema,
    db: AsyncSession = Depends(get_async_db),
    _: UUID = Depends(get_current_user),
    qdrant: AsyncQdrantClient = Depends(get_qdrant_client),
):
    service = RetrieveEmbeddingsService(qdrant, db)
    return await service.test_retrieval(
        chatbot_id=chatbot_id,
        payload=payload,
    )
    