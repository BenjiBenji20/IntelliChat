from uuid import UUID
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------
ALLOWED_MIME_TYPES: set[str] = {
    "application/pdf",
    "text/plain",
    "text/markdown",
    "application/json",
    "application/jsonl",
}

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB


class GenerateUploadURLRequestSchema(BaseModel):
    chatbot_id: UUID
    file_name: str = Field(..., min_length=1, max_length=255)
    file_type: str = Field(..., description="MIME type of the file")
    file_size: int = Field(..., gt=0, description="File size in bytes")

    def validate_file_type(self) -> None:
        if self.file_type not in ALLOWED_MIME_TYPES:
            raise ValueError(f"Unsupported file type: {self.file_type}")

    def validate_file_size(self) -> None:
        if self.file_size > MAX_FILE_SIZE_BYTES:
            raise ValueError(
                f"File size {self.file_size} exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES} bytes."
            )
    

class BulkUploadURLRequestSchema(BaseModel):
    chatbot_id: UUID
    files: list[GenerateUploadURLRequestSchema] = Field(..., min_length=1, max_length=10)


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------
class GenerateUploadURLResponseSchema(BaseModel):
    document_id: UUID
    upload_url: str
    storage_path: str
    expires_in_seconds: int


class ConfirmUploadResponseSchema(BaseModel):
    document_id: UUID
    chatbot_id: UUID
    status: str
    message: str


class DownloadURLResponseSchema(BaseModel):
    document_id: UUID
    download_url: str
    expires_in_seconds: int


class DocumentStatusResponseSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    document_id: UUID = Field(validation_alias="id")
    file_name: str
    file_type: str
    file_size: int
    status: str
    storage_path: str
    created_at: datetime
    updated_at: datetime


class DeleteDocumentResponseSchema(BaseModel):
    file_name: str
    message: str
    

class BulkUploadURLResponseSchema(BaseModel):
    results: list[GenerateUploadURLResponseSchema]
    total: int
    failed: list[str] = []  # file_names that failed validation


class DocumentConfigurationRequestSchema(BaseModel):
    document_id: UUID
    # Applied to .pdf, .txt, .md
    chunk_size: int = Field(default=500, gt=0)
    chunk_overlap: int = Field(default=50, ge=0)
    separator: str = Field(default="\n\n")
    # Applied to all file types
    document_type: str = Field(default="knowledge_base")


class BulkConfirmRequestSchema(BaseModel):
    chatbot_id: UUID
    document_ids: list[UUID] = Field(..., min_length=1, max_length=10)
    document_configurations: list[DocumentConfigurationRequestSchema] = Field(default_factory=list)

class BulkConfirmResponseSchema(BaseModel):
    confirmed: list[ConfirmUploadResponseSchema]
    failed: list[dict]  # {"document_id": ..., "reason": ...}


class DocumentListResponseSchema(BaseModel):
    documents: list[DocumentStatusResponseSchema]
    total: int
    limit: int
    offset: int
    chatbot_id: UUID
    
    
# for serialization
class ProcessDocumentRequestSchema(BaseModel):
    document_id: UUID
    chatbot_id: UUID
    file_name: str
    file_type: str  # txt, md, json, jsonl, pdf
    
    # optional chunking configuration of: txt, pdf files
    document_type: str = "knowledge_base"
    chunk_size: int = 500
    chunk_overlap: int = 50
    separator: str = "\n\n"
    