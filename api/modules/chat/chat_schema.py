import re
from datetime import datetime
from uuid import UUID
 
from pydantic import BaseModel, Field, field_validator
 
from api.modules.retrievals.retrieval_schema import ChunkResultSchema, RetrievalFilter

class IntelliChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500, description="User's message")
    @field_validator("query")
    def validate_query(cls, v: str):
        v = v.strip()
        special_char_ratio = sum(not c.isalnum() for c in v) / len(v)
        v = re.sub(r"([?!.,;:])\1+", r"\1", v)
        if special_char_ratio > 0.5:
            v = v.rstrip("?!.,;:")
        return v
    filters: list[RetrievalFilter] = Field(default_factory=list)
    conversation_id: str
    top_k: int = Field(default=5, gt=0, le=20)
 
 
class UserResponse(BaseModel):
    role: str = "user"
    query: str
    created_at: datetime
 
 
class AssistantResponse(BaseModel):
    role: str = "assistant"
    content: str | None = None
    created_at: datetime
 
 
class UsageResponse(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
 
 
class ModelMetadataResponse(BaseModel):
    # LLM metadata — always required
    llm_model: str
    llm_provider: str
    temperature: float = 0.70
 
    # Embedding metadata — optional (absent when no retriever is used)
    embedding_model_name: str | None = None
    embedding_model_provider: str | None = None
    # vector_size: int | None = None
    top_k: int | None = None
 
 
class IntellichatResponseSchema(BaseModel):
    id: UUID
    conversation_id: str
    chatbot_id: UUID
    client: UserResponse
    assistant: AssistantResponse
    sources: list[ChunkResultSchema] = Field(default_factory=list)    # empty list when no retriever
    usage: UsageResponse
    model_metadata: ModelMetadataResponse