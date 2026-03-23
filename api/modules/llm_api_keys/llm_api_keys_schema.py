from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from uuid import UUID
from datetime import datetime
    
class CreateRequestLlmSchema(BaseModel):
    user_id: UUID
    chatbot_id: UUID
    project_id: UUID
    api_key: str
    llm_name: str = Field(..., max_length=100)
    temperature: float = 0.70
    provider: str
    
    @field_validator('llm_name', 'provider')
    @classmethod
    def normalize(cls, v: str) -> str:
        return v.lower().strip()
    
class ResponseLlmSchema(CreateRequestLlmSchema):
    id: UUID
    created_at: datetime
    updated_at: datetime


class UpdateRequestLlmSchema(BaseModel):
    project_id: UUID
    
    # new fields - to update old fields
    new_raw_api_key: str | None = None        # client sending NEW raw key
    new_llm_name: str | None = None
    new_provider: str | None = None
    new_temperature: float | None = None
    
    # old fields - all required
    old_encrypted_api_key: str   # client sending EXISTING encrypted key back
    old_llm_name: str = Field(..., max_length=100)
    old_provider: str 
    old_temperature: float
    
    @field_validator('new_llm_name', 'new_provider', 'old_llm_name', 'old_provider')
    @classmethod
    def normalize(cls, v: str) -> str:
        return v.lower().strip()
