from datetime import datetime

from pydantic import BaseModel, Field
from uuid import UUID

class ChatbotStateSchema(BaseModel):
    chatbot_id: UUID | None = None
    chatbot_completed: bool
    llm_completed: bool
    embedding_completed: bool
    chatbot_data: dict | None = None
    llm_data: dict | None = None
    embedding_data: dict | None = None


class CreateRequestChatbotSchema(BaseModel):
    user_id: UUID
    project_id: UUID
    application_name: str = Field(..., max_length=100)
    has_memory: bool = False
    system_prompt: str | None = None
    
class UpdateRequestChatbotSchema(BaseModel):
    project_id: UUID
    application_name: str = Field(None, max_length=100)
    has_memory: bool | None = None
    system_prompt: str | None = None
    
class ResponseChatbotSchema(CreateRequestChatbotSchema):
    id: UUID
    user_id: UUID
    is_active: bool = True
    created_at: datetime
    updated_at: datetime
    
    
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
    
    
class CreateRequestEmbbedingModelSchema(BaseModel):
    user_id: UUID
    chatbot_id: UUID
    api_key: str
    embedding_model_name: str = Field(..., max_length=100)
    provider: str
    
class UpdateRequestEmbeddingModelSchema(BaseModel):
    project_id: UUID
    api_key: str | None = None
    embedding_model_name: str = Field(None, max_length=100)
    provider: str | None = None
    
class ResponseEmbbedingModelSchema(CreateRequestEmbbedingModelSchema):
    id: UUID
    user_id: UUID
    created_at: datetime
    updated_at: datetime 
    