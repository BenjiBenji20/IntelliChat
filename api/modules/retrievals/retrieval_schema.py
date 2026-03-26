import re
from pydantic import BaseModel, Field, field_validator

class RetrievalFilter(BaseModel):
    file_type: str | None = None # file extension ex. txt
    @field_validator('file_type')
    @classmethod
    def normalize_file_type(cls, v: str):
        if v is None:
            return None
        return v.lstrip('.').lower().strip()  # ".PDF" to "pdf"
    
    file_name: str | None = None # actual filename
    title: str | None = None # file title in metadata
    content_type: str | None = None
    document_type: str | None = None
    
    # Document-type specific metadata fields
    page_number: int | None = None
    section: str | None = None
    heading_level: int | None = None
    json_path: str | None = None
    record_id: str | None = None

class RetrievalRequestSchema(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=500,
        description="User's message"
    )
    filters: list[RetrievalFilter] = Field(default_factory=list)
    top_k: int = Field(default=5, gt=0, le=20)
    
    @field_validator('query')
    def validate_query(cls, v: str):
        v = v.strip()
        # Calculate special character ratio
        special_char_ratio = sum(not c.isalnum() for c in v) / len(v)
        # Collapse repeated punctuation
        v = re.sub(r'([?!.,;:])\1+', r'\1', v)

        # remove trailing punctuation if too noisy
        if special_char_ratio > 0.5:
            v = v.rstrip('?!.,;:')

        return v
    

class ChunkResultSchema(BaseModel):
    document_id: str
    chunk_id: str
    chunk_index: int
    score: float
    page_content: str
    file_name: str
    file_type: str
    content_type: str
    document_type: str
    ingestion_time: str
    # optional — only present for specific file types
    page_number: int | None = None
    section: str | None = None
    heading_level: int | None = None
    json_path: str | None = None
    record_id: str | None = None


class RetrievalResponseSchema(BaseModel):
    query: str
    top_k: int
    total_results: int
    results: list[ChunkResultSchema]


class CollectionStatsSchema(BaseModel):
    model_name: str
    model_size: int
    model_distance: str
    total_documents: int
    storage_kb: int
    