from datetime import datetime
from pydantic import BaseModel, Field
from uuid import UUID
from datetime import datetime
    
class CreateRequestLlmSchema(BaseModel):
    user_id: UUID
    chatbot_id: UUID
    api_key: str
    llm_name: str = Field(..., max_length=100)
    temperature: float = 0.70
    provider: str

class UpdateRequestLlmSchema(BaseModel):
    project_id: UUID
    api_key: str | None = None
    llm_name: str = Field(None, max_length=100)
    temperature: float | None = None
    provider: str | None = None

class ResponseLlmSchema(CreateRequestLlmSchema):
    id: UUID
    user_id: UUID
    created_at: datetime
    updated_at: datetime
    