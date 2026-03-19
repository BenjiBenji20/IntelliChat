import logging

from asyncpg import Connection
from cryptography.fernet import Fernet, InvalidToken
from fastapi import HTTPException
import asyncio

from doc_worker.configs.settings import settings
from doc_worker.configs.gcs import get_gcs_bucket
from doc_worker.modules.documents.document_worker_schema import *
from doc_worker.modules.documents.document_worker_repository import DocumentWorkerRepository
from doc_worker.modules.chunkings.load_file import file_loader
from doc_worker.modules.chunkings.chunker_factory import chunker_factory
from doc_worker.modules.embeddings.embedder_factory import EmbedderFactory
from doc_worker.modules.qdrant.qdrant_service import qdrant
from shared.gcs_file_path import construct_file_path

logger = logging.getLogger(__name__)

class DocumentWorkerService:
    def __init__(self, db: Connection):
        self.db = db
        self.repo = DocumentWorkerRepository(db)
        
        self.ALLOWED_FILE_TYPES = {"txt", "md", "json", "jsonl", "pdf"}


    async def process_document(
        self, payload: ProcessDocumentRequestSchema
    ) -> ProcessDocumentResponseSchema:
        """
        Update document status failed on failure
        IMMEDIATE RETURN on File, embedding model details not found to response 2XX and prevent cloud tasks retry
        Every successfull process updates the document status
        Final success return as document status = indexed
        """
        try:
            # check document status by id 
            doc_status = await self.repo.get_document_status(payload.document_id)

            if not doc_status:
                return ProcessDocumentResponseSchema(
                    message=f"Document with ID: {payload.document_id} not found in database.",
                    document_id=payload.document_id,
                    status="failed"
                )
            
            # if status = indexed skip this tasks to avoid double embeddings of same file
            if doc_status == "indexed":
                return ProcessDocumentResponseSchema(
                    message="Document already indexed.",
                    document_id=payload.document_id,
                    status="indexed"
                )
            
            # Validate file type first before any DB call
            if payload.file_type.strip() not in self.ALLOWED_FILE_TYPES:
                # update return immediately on unsupport file
                await self.repo.update_document_status(payload.document_id, "failed")
                return ProcessDocumentResponseSchema(
                    message="Unsupported file type",
                    document_id=payload.document_id,
                    status="failed"
                )

            # Update status to processing
            await self.repo.update_document_status(payload.document_id, "processing")

            # Fetch embedding key
            model_details = await self.repo.get_embedding_model_details(payload.chatbot_id)
            if not model_details or not model_details["api_key_encrypted"] or not model_details["embedding_model_name"]:
                await self.repo.update_document_status(payload.document_id, "failed")
                # immediate return on model details not found to avoid retry
                return ProcessDocumentResponseSchema(
                    message=(
                        "Embedding model and API key not found."
                        "Configure your chatbot project first before document processing."
                    ),
                    document_id=payload.document_id,
                    status="failed"
                )
                
            # decrypt api key and update model_details
            raw_key = self.decrypt_secret(model_details["api_key_encrypted"])
            if not raw_key:
                await self.repo.update_document_status(payload.document_id, "failed")
                return ProcessDocumentResponseSchema(
                    message="Failed to decrypt API key.",
                    document_id=payload.document_id,
                    status="failed"
                )
                
            model_details.update({"api_key_raw": raw_key})
            del model_details["api_key_encrypted"] # delete encrypted key

            # Download file from GCS
            file_path = construct_file_path(
                payload.chatbot_id,
                payload.document_id,
                payload.file_name
            )
            
            dl_file_bytes: bytes = await self.download_file_from_gcs(file_path)
            if not dl_file_bytes:
                await self.repo.update_document_status(payload.document_id, "failed")
                # immediate return on file not found to avoid retry
                return ProcessDocumentResponseSchema(
                    message="File not found.",
                    document_id=payload.document_id,
                    status="failed"
                )

            # Load file bytes into list[dict] or string
            if payload.file_type == "pdf": # skip loading for pdf:
                loaded_file = None # pdf chunker handles raw bytes directly
            else:
                loaded_file = await file_loader.load(dl_file_bytes, payload.file_type)
                if not loaded_file:
                    await self.repo.update_document_status(payload.document_id, "failed")
                    return ProcessDocumentResponseSchema(
                        message="Failed to load file content",
                        document_id=payload.document_id,
                        status="failed"
                    )
                
            # Chunk files into documents good for embedding 
            try:
                chunk_file = chunker_factory.get_chunker(
                    payload.file_type,
                    payload.document_type,
                    payload.chunk_size,
                    payload.chunk_overlap
                )
            except ValueError as e:
                await self.repo.update_document_status(payload.document_id, "failed")
                return ProcessDocumentResponseSchema(
                    message=f"Failed to chunk document due to unsupported file type: {payload.file_type}",
                    document_id=payload.document_id,
                    status="failed"
                )
            
            # pass the bytes for pdf files
            if payload.file_type == "pdf":
                document_chunks = await asyncio.to_thread(
                    chunk_file.chunk,
                    dl_file_bytes,
                    payload.document_id,
                    payload.file_name
                )
            else:
                document_chunks = await asyncio.to_thread(
                    chunk_file.chunk,
                    loaded_file,
                    payload.document_id,
                    payload.file_name
                )
                
            if not document_chunks:
                await self.repo.update_document_status(payload.document_id, "failed")
                return ProcessDocumentResponseSchema(
                    message=f"Failed to chunk {payload.file_type.capitalize()} document.",
                    document_id=payload.document_id,
                    status="failed"
                )
                
            # Embed each document into vector floats
            try:
                # create embedder model using the factory to get the proper library based on model and provider
                embedder = EmbedderFactory.get_embedder(
                    provider=model_details["provider"],
                    api_key=model_details["api_key_raw"],
                    model_name=model_details["embedding_model_name"]
                )
            except ValueError:
                await self.repo.update_document_status(payload.document_id, "failed")
                return ProcessDocumentResponseSchema(
                    message=f"Failed to make vector embeddings for the document: {payload.file_name}",
                    document_id=payload.document_id,
                    status="failed"
                )
            
            # embed all chunks
            vectors = []
            for chunk in document_chunks:
                vector = await embedder.embed(chunk.page_content)
                if vector is None:
                    await self.repo.update_document_status(payload.document_id, "failed")
                    return ProcessDocumentResponseSchema(
                        message="Failed to embed document chunks",
                        document_id=payload.document_id,
                        status="failed"
                    )
                vectors.append(vector)
            
            # store vectors in Qdrant
            # Qdrant — ensure collection + delete old + upsert
            # errors raise by try-except wrapper
            try:
                await qdrant.ensure_collection_exists(
                    chatbot_id=payload.chatbot_id,
                    vector_name=embedder.VECTOR_NAME,
                    vector_size=embedder.VECTOR_SIZE
                )
                await qdrant.delete_document_vectors(payload.chatbot_id, payload.document_id)
                await qdrant.upsert_document_chunks(
                    vector_name=embedder.VECTOR_NAME,
                    chatbot_id=payload.chatbot_id,
                    document_id=payload.document_id,
                    chunks=document_chunks,
                    vectors=vectors
                )
            except Exception as e:
                logger.error(
                    f"Qdrant operation failed for "
                    f"document_id={payload.document_id}. Error: {e}"
                )
                raise  # Cloud Tasks retries
            
            # status = "indexed" after all steps completed
            await self.repo.update_document_status(payload.document_id, "indexed")
            return ProcessDocumentResponseSchema(
                message="Document successfully indexed. Ready for document retrieval!",
                document_id=payload.document_id,
                status="indexed"
            )

        except HTTPException:
            raise
        except Exception as e:
            # Recoverable error and let Cloud Tasks retry
            raise e    

        
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    async def download_file_from_gcs(self, file_path: str) -> bytes | None:
        bucket = get_gcs_bucket()
        blob = bucket.blob(file_path)
        
        exists = await asyncio.get_running_loop().run_in_executor(None, blob.exists)
        if not exists:
            return None
        
        return await asyncio.get_running_loop().run_in_executor(None, blob.download_as_bytes)
    
    
    def decrypt_secret(self, encrypted_key: str) -> str | None:
        """
        Decrypt an encrypted secret key string.
        Return None if invalid or tampered with.
        """
        try:
            if not settings.ENCRYPTION_KEY:
                logger.exception("Missing encryption key")
                return None
            
            cipher_suite = Fernet(settings.ENCRYPTION_KEY.encode())
            return cipher_suite.decrypt(encrypted_key.encode()).decode()
        except InvalidToken:
            logger.exception("Invalid or corrupted encryption key")
            return None
