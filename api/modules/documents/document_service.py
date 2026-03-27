import asyncio
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.cache.redis_service import FREQ_CACHE_PREFIX, redis_service
from api.configs.settings import settings
from api.models.document import Document
from api.modules.documents.document_repository import DocumentRepository
from api.modules.documents.chunking_config_repository import ChunkingConfigurationRepository
from api.modules.documents.document_schema import *
from api.modules.documents.gcs_service import (
    SIGNED_URL_UPLOAD_EXPIRY_SECONDS,  # re-exported for response metadata
    gcs_service,
)
from shared.gcs_file_path import construct_file_path
from api.modules.qdrant.qdrant_repository import qdrant_repo
import logging

_DEFAULT_CONFIG = {
    "chunk_size": 500,
    "chunk_overlap": 50,
    "separator": "\n\n",
    "document_type": "knowledge_base",
}

logger = logging.getLogger(__name__)

class DocumentService:
    """
    Orchestrates GCS operations and DB persistence.
    GCS calls are offloaded to a thread pool (asyncio.to_thread) because
    the google-cloud-storage SDK is synchronous.
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.document_repo = DocumentRepository(db)
        self.chunking_config_repo = ChunkingConfigurationRepository(db)


    # ------------------------------------------------------------------
    # Generate signed download URL
    # ------------------------------------------------------------------
    async def generate_download_url(
        self,
        chatbot_id: UUID,
        document_id: UUID,
    ) -> DownloadURLResponseSchema:
        try:
            document = await self._get_owned_document(document_id, chatbot_id)

            if document.status == "pending":
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="File upload has not been confirmed yet.",
                )

            download_url = await asyncio.to_thread(
                gcs_service.generate_signed_download_url,
                document.storage_path,
            )

            return DownloadURLResponseSchema(
                document_id=document_id,
                download_url=download_url,
                expires_in_seconds=SIGNED_URL_UPLOAD_EXPIRY_SECONDS,
            )
            
        except HTTPException:
            raise
        
        except Exception as e:
            await self.db.rollback()
            raise e


    # ------------------------------------------------------------------
    # Delete document
    # ------------------------------------------------------------------
    async def delete_document(
        self,
        chatbot_id: UUID,
        document_id: UUID,
    ) -> DeleteDocumentResponseSchema:
        try:
            document = await self._get_owned_document(document_id, chatbot_id)

            # Delete from GCS first — if this fails, DB record is still intact
            await asyncio.to_thread(gcs_service.delete_object, document.storage_path)

            # Then delete DB record (cascades to embeddings_metadata via FK)
            await self.document_repo.delete(document.id)

            # Delete document's vectors in qdrant
            await qdrant_repo.delete_document_vectors(
                chatbot_id=chatbot_id,
                document_id=document_id
            )
            
            # delete current cached collection stats
            # to update it by requesting new one
            is_deleted = await redis_service.delete(
                key=str(chatbot_id), prefix=f"{FREQ_CACHE_PREFIX}(collection_stats)"
            )
            info_message = "is DELETED." if is_deleted else "is FAILED to delete."
            logger.info(
                f"[INFO] CACHE: key={str(chatbot_id)} prefix={FREQ_CACHE_PREFIX}(collection_stats) "
                f"{info_message}"
            )
    
            return DeleteDocumentResponseSchema(
                file_name=document.file_name,
                message="Document and all associated data deleted successfully.",
            )
                    
        except HTTPException:
            raise
        
        except Exception as e:
            await self.db.rollback()
            raise e


    # ------------------------------------------------------------------
    # Bulk generate signed upload URLs
    # ------------------------------------------------------------------
    async def bulk_generate_upload_urls(
        self,
        user_id: UUID,
        payload: BulkUploadURLRequestSchema,
    ) -> BulkUploadURLResponseSchema:
        failed = []
        valid_files: list[Document] = []

        # Validate all files upfront before touching DB or GCS
        for file in payload.files:
            try:
                file.validate_file_type()
                file.validate_file_size()
                valid_files.append(file)
            except ValueError as exc:
                failed.append(file.file_name)

        if not valid_files:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="File types are invalid.",
            )

        try:
            # Single INSERT for all valid files
            documents = await self.document_repo.bulk_create_documents(
                user_id=user_id,
                chatbot_id=payload.chatbot_id,
                files=[
                    {
                        "file_name": f.file_name, 
                        "file_type": f.file_type,
                        "file_size": f.file_size,
                        "storage_path": "", 
                        "status": "pending"
                    }
                    for f in valid_files
                ],
            )

            # Build storage paths now that we have document IDs
            results = []
            for document, file in zip(documents, valid_files):
                storage_path = construct_file_path(
                    payload.chatbot_id, document.id, file.file_name
                )
                document.storage_path = storage_path

                # GCS signed URL — still per-file (each URL is unique per object key)
                # asyncio.gather here is safe since these are pure HTTP calls with no
                # shared DB state — no session risk
                upload_url = await asyncio.to_thread(
                    gcs_service.generate_signed_upload_url,
                    storage_path,
                    file.file_type,
                    file.file_size,
                )

                results.append(
                    GenerateUploadURLResponseSchema(
                        document_id=document.id,
                        upload_url=upload_url,
                        storage_path=storage_path,
                        expires_in_seconds=SIGNED_URL_UPLOAD_EXPIRY_SECONDS,
                    )
                )

            return BulkUploadURLResponseSchema(
                results=results,
                total=len(results),
                failed=failed,
            )

        except HTTPException:
            raise
        except Exception as e:
            await self.db.rollback()
            raise e


    # ------------------------------------------------------------------
    # Bulk confirm uploads
    # ------------------------------------------------------------------
    async def bulk_confirm_uploads(
        self,
        chatbot_id: UUID,
        payload: BulkConfirmRequestSchema,
    ) -> tuple[BulkConfirmResponseSchema, list[dict]]:
        """
        Confirms GCS uploads, bulk-updates status to 'uploaded',
        then enqueues one Cloud Task per document (non-blocking).
    
        Ownership is enforced by scoping the DB fetch to chatbot_id —
        any document_id not belonging to this chatbot is silently moved
        to the failed list.
        """
        try:
            # Single IN query for all requested document IDs
            documents = await self.document_repo.get_by_ids_and_chatbot_id(
                payload.document_ids, chatbot_id
            )

            doc_map = {doc.id: doc for doc in documents}
            confirmed_ids = []
            confirmed_responses = []
            task_payloads = []
            failed = []
            
            # Build config lookup: document_id → config dict
            config_map: dict[str, dict] = {
                str(cfg.document_id): {
                    "chunk_size": cfg.chunk_size,
                    "chunk_overlap": cfg.chunk_overlap,
                    "separator": cfg.separator,
                    "document_type": cfg.document_type,
                }
                for cfg in payload.document_configurations
            }

            # GCS existence checks — these are I/O bound so gather them in parallel
            async def check_exists(doc: Document):
                return doc.id, await asyncio.to_thread(
                    gcs_service.object_exists, doc.storage_path
                )

            # Only check docs that are actually pending
            pending_docs = [
                doc_map[doc_id]
                for doc_id in payload.document_ids
                if doc_map.get(doc_id) and doc_map[doc_id].status == "pending"
            ]

            # Surface not-found and wrong-status errors first
            for doc_id in payload.document_ids:
                doc = doc_map.get(doc_id)
                if not doc:
                    failed.append({
                        "document_id": str(doc_id), 
                        "reason": "Document not found."
                    })
                elif doc.status != "pending":
                    failed.append({
                        "document_id": str(doc_id), 
                        "reason": f"Already in status '{doc.status}'."
                    })

            # Parallel GCS existence checks for all pending docs
            existence_results = await asyncio.gather(
                *[check_exists(doc) for doc in pending_docs],
                return_exceptions=True,
            )

            for result in existence_results:
                if isinstance(result, Exception):
                    # Can't identify which doc failed from exception alone; log and skip
                    continue
                doc_id, exists = result
                doc = doc_map[doc_id]
                
                if not exists:
                    # File never made it to GCS — clean up DB record
                    await self.document_repo.delete(doc_id)
                    failed.append({
                        "document_id": str(doc_id), 
                        "reason": "File not found in storage. Upload may have failed."
                    })
                    continue
                
                # File exists — check size
                metadata = await asyncio.to_thread(
                    gcs_service.get_object_metadata, doc.storage_path
                )
                    
                # Remove the files larger than max file size. Extract size in object metadata and remove in GCS
                if metadata and metadata["size"] > settings.MAX_FILE_SIZE_BYTES:
                    await asyncio.to_thread(
                        gcs_service.delete_object, 
                        doc.storage_path
                    )
                    
                    await self.document_repo.update_status(doc_id, "failed") # status as failed
                    
                    failed.append({
                        "document_id": str(doc_id), 
                        "reason": f"File size exceeds maximum allowed {settings.MAX_FILE_SIZE_BYTES} bytes."
                    })
                    
                    continue
                
                # All checks passed — collect for bulk update and task enqueue
                doc_id_str = str(doc_id)
                cfg = config_map.get(doc_id_str, _DEFAULT_CONFIG)
                
                confirmed_ids.append(doc_id)
                confirmed_responses.append(
                    ConfirmUploadResponseSchema(
                        document_id=doc_id,
                        chatbot_id=chatbot_id,
                        status="uploaded",
                        message="Upload confirmed. Document is queued for processing.",
                    )
                )
                
                file_extension = doc.file_name.rsplit(".", 1)[-1].lower()  # "file.txt" → "txt"
                
                task_payloads.append({
                    "document_id": doc_id_str,
                    "chatbot_id": str(chatbot_id),
                    "file_name": doc.file_name,
                    "file_type": file_extension,
                    "chunk_size": cfg["chunk_size"],
                    "chunk_overlap": cfg["chunk_overlap"],
                    "separator": cfg["separator"],
                    "document_type": cfg["document_type"],
                })
                
                chunk_configs_to_create = []
                chunk_configs_to_create.append({
                    "document_id": doc_id,
                    "chatbot_id": str(chatbot_id),
                    "chunk_size": cfg["chunk_size"],
                    "chunk_overlap": cfg["chunk_overlap"],
                    "separator": cfg["separator"],
                    "document_type": cfg["document_type"],
                    "content_type": cfg.get("content_type", "knowledge") # hardcoded for now
                })

            # Single UPDATE ... WHERE id IN (...) for all confirmed docs
            if confirmed_ids:
                await self.document_repo.bulk_update_status(confirmed_ids, "uploaded")
                
                for chunk_config in chunk_configs_to_create:
                    await self.chunking_config_repo.create(**chunk_config)
                    
                # delete current cached collection stats
                # to update it by requesting new one
                is_deleted = await redis_service.delete(
                    key=str(chatbot_id), prefix=f"{FREQ_CACHE_PREFIX}(collection_stats)"
                )
                info_message = "is DELETED." if is_deleted else "is FAILED to delete."
                logger.info(
                    f"[INFO] CACHE: key={str(chatbot_id)} prefix={FREQ_CACHE_PREFIX}(collection_stats) "
                    f"{info_message}"
                )
                
            return BulkConfirmResponseSchema(confirmed=confirmed_responses, failed=failed), task_payloads

        except HTTPException:
            raise
        except Exception as e:
            await self.db.rollback()
            raise e


    # ------------------------------------------------------------------
    # List documents by chatbot (paginated)
    # ------------------------------------------------------------------
    async def list_documents_by_chatbot(
        self,
        chatbot_id: UUID,
        limit: int = 20,
        offset: int = 0,
    ) -> DocumentListResponseSchema:
        documents, total = await self.document_repo.get_all_by_chatbot_id(
            chatbot_id, limit=limit, offset=offset
        )
        return DocumentListResponseSchema(
            documents=[DocumentStatusResponseSchema.model_validate(d) for d in documents],
            total=total,
            limit=limit,
            offset=offset,
            chatbot_id=chatbot_id,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    async def _get_owned_document(
        self, document_id: UUID, chatbot_id: UUID
    ) -> Document:
        document = await self.document_repo.get_by_document_and_chatbot_id(
            document_id, chatbot_id
        )
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found.",
            )
            
        return document
