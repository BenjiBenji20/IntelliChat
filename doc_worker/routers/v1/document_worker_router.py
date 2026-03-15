from fastapi import APIRouter, status, Depends, Header, HTTPException
from asyncpg import Connection

from doc_worker.db.db_session import get_async_db
from doc_worker.modules.documents.document_worker_schema import *
from doc_worker.configs.settings import settings
from doc_worker.modules.documents.document_worker_service import DocumentWorkerService

router = APIRouter(
    prefix="/worker/document",
    tags=["Document Processing Worker for Intellichat Embeddings"]
)


# In File dependency
async def verify_cloud_tasks(
    x_cloudtasks_queuename: str = Header(None),
    x_cloudtasks_taskname: str = Header(None)
):
    """Headers automatically set by Cloud tasks"""
    if not x_cloudtasks_queuename or not x_cloudtasks_taskname:
        raise HTTPException(status_code=403, detail="Forbidden")
    if x_cloudtasks_queuename != settings.QUEUE_NAME:
        raise HTTPException(status_code=403, detail="Forbidden")


@router.post(
    "/process",  
    status_code=status.HTTP_202_ACCEPTED,
    response_model=ProcessDocumentResponseSchema,
    summary="Process documents from main API server",
    dependencies=[Depends(verify_cloud_tasks)]
)
async def process_document(
    payload: ProcessDocumentRequestSchema,
    db: Connection = Depends(get_async_db)
):
    service = DocumentWorkerService(db)
    return await service.process_document(payload=payload)
