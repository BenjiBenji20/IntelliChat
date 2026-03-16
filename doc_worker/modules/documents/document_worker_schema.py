from pydantic import BaseModel
from uuid import UUID

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

    
class ProcessDocumentResponseSchema(BaseModel):
    message: str
    document_id: UUID
    status: str
    