from datetime import datetime
from pydantic import BaseModel, Field
from uuid import UUID
from datetime import datetime
    
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
    