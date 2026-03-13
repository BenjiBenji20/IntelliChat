from pydantic import BaseModel, Field
from uuid import UUID
from datetime import datetime
from typing import Optional


class BaseProjectSchema(BaseModel):
    name: str = Field(..., max_length=100)
    description: Optional[str] = None
    is_active: bool = True


class CreateProjectSchema(BaseProjectSchema):
    pass


class UpdateProjectSchema(BaseModel):
    name: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = None
    is_active: Optional[bool] = None


class ResponseProjectSchema(BaseProjectSchema):
    id: UUID
    owner_id: UUID
    is_active: bool
    created_at: datetime
    updated_at: datetime
    
    api_key: Optional[str] = Field(None, max_length=100) # will be responsed once

    model_config = {
        "from_attributes": True
    }


class ProjectPatchSchema(BaseModel):
    name: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = None
    is_active: Optional[bool] = None
