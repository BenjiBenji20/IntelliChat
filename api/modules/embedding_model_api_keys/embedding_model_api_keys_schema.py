from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from uuid import UUID
from datetime import datetime
    
class CreateRequestEmbbedingModelSchema(BaseModel):
    user_id: UUID
    project_id: UUID
    chatbot_id: UUID
    api_key: str
    embedding_model_name: str = Field(..., max_length=100)
    provider: str
    
    @field_validator('embedding_model_name', 'provider')
    @classmethod
    def normalize(cls, v: str) -> str:
        return v.lower().strip()
    
class ResponseEmbbedingModelSchema(CreateRequestEmbbedingModelSchema):
    id: UUID
    created_at: datetime
    updated_at: datetime 
    
    
class UpdateRequestEmbeddingModelSchema(BaseModel):
    project_id: UUID
    
    # new fields - to update old fields
    new_raw_api_key: str | None = None        # client sending NEW raw key
    new_embedding_model_name: str | None = None
    new_provider: str | None = None
    new_temperature: float | None = None

    # old fields - all required    
    old_encrypted_api_key: str   # client sending EXISTING encrypted key back
    old_embedding_model_name: str = Field(..., max_length=100)
    old_provider: str
    
    @field_validator('new_embedding_model_name', 'new_provider', 'old_embedding_model_name', 'old_provider')
    @classmethod
    def normalize(cls, v: str) -> str:
        return v.lower().strip()
    