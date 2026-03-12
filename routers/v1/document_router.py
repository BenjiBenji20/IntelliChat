from uuid import UUID

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from db.db_session import get_async_db
from dependencies.auth import get_current_user 
from modules.documents.document_schema import *
from modules.documents.document_service import DocumentService
from dependencies.rate_limit import rate_limit_by_user

router = APIRouter(
    prefix="/api/documents", 
    tags=["Document Storage"]
)

@router.get(
    "/download/{chatbot_id}/{document_id}",
    response_model=DownloadURLResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Generate a signed download URL for a document",
    dependencies=[Depends(rate_limit_by_user())]
)
async def get_download_url(
    chatbot_id: UUID,
    document_id: UUID,
    _: UUID = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    service = DocumentService(db)
    return await service.generate_download_url(
        chatbot_id=chatbot_id,
        document_id=document_id
    )


@router.delete(
    "/delete/{chatbot_id}/{document_id}",
    response_model=DeleteDocumentResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Delete a document from GCS and the database",
    description=(
        "Deletes the file from GCS first, then removes the DB record. "
        "Cascades to embeddings_metadata via foreign key."
    ),
    dependencies=[Depends(rate_limit_by_user())]
)
async def delete_document(
    chatbot_id: UUID,
    document_id: UUID,
    _: UUID = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    service = DocumentService(db)
    return await service.delete_document(
        chatbot_id=chatbot_id,
        document_id=document_id
    )


@router.post(
    "/bulk-upload-url",
    response_model=BulkUploadURLResponseSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Generate signed upload URLs for multiple files at once",
    description=(
        "Accepts up to 10 files. Returns signed PUT URLs for each. "
        "Files that fail validation are listed in 'failed' and do not block the rest. "
        "Call /bulk-confirm after all PUTs complete."
    ),
    dependencies=[Depends(rate_limit_by_user())]
)
async def bulk_generate_upload_urls(
    payload: BulkUploadURLRequestSchema,
    current_user_id: UUID = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    service = DocumentService(db)
    return await service.bulk_generate_upload_urls(
        user_id=current_user_id,
        payload=payload,
    )


@router.post(
    "/bulk-confirm",
    response_model=BulkConfirmResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Confirm multiple uploads at once",
    description=(
        "Call after all GCS PUTs complete. "
        "Each document is confirmed independently — partial success is valid. "
        "Check 'failed' list for any that didn't land."
    ),
    dependencies=[Depends(rate_limit_by_user())]
)
async def bulk_confirm_uploads(
    payload: BulkConfirmRequestSchema,
    _: UUID = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    service = DocumentService(db)
    return await service.bulk_confirm_uploads(
        chatbot_id=payload.chatbot_id,
        payload=payload,
    )


@router.get(
    "/",
    response_model=DocumentListResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="List all documents and statuses for a chatbot (paginated)",
    dependencies=[Depends(rate_limit_by_user())]
)
async def list_documents(
    chatbot_id: UUID,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    _: UUID = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    service = DocumentService(db)
    return await service.list_documents_by_chatbot(
        chatbot_id=chatbot_id,
        limit=limit,
        offset=offset,
    )
    