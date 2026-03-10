from datetime import datetime
from uuid import UUID
import re
from pydantic import BaseModel, Field, field_validator

# ============================================================
# ALLOWED CHARACTERS PATTERN
# Allows letters, numbers, spaces, and safe punctuation only
# Blocks: < > & " ' ; { } [ ] ` \ / | ^ ~ 
# ============================================================
SAFE_TEXT_PATTERN = re.compile(r'^[a-zA-Z0-9\s.,!?()\-_:\'\"@#%+=]*$')
SAFE_SHORT_PATTERN = re.compile(r'^[a-zA-Z0-9\s\-_,.()/]*$')  # stricter for short fields

def sanitize_text(value: str | None, field_name: str, strict: bool = False) -> str | None:
    if value is None:
        return None
    
    value = value.strip()
    
    if not value:
        return None
    
    pattern = SAFE_SHORT_PATTERN if strict else SAFE_TEXT_PATTERN
    if not pattern.match(value):
        raise ValueError(
            f"'{field_name}' contains invalid characters. "
            f"Special characters like <, >, &, ;, {{, }}, `, \\, | are not allowed."
        )
    
    # Block common prompt injection triggers
    INJECTION_PATTERNS = [
        r'ignore\s+(all\s+)?(previous|prior|above)\s+instructions?',
        r'you\s+are\s+now',
        r'act\s+as\s+(a\s+)?(?!if)',
        r'disregard\s+(all\s+)?instructions?',
        r'system\s*:\s*',
        r'<\s*system\s*>',
        r'\[INST\]',
        r'###\s*instruction',
    ]
    
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, value, re.IGNORECASE):
            raise ValueError(
                f"'{field_name}' contains potentially unsafe content."
            )
    
    return value


class BaseBehaviorStudioSchema(BaseModel):
    category: str | None = Field(None, max_length=255)
    target_audience: str | None = Field(None, max_length=255)
    description: str | None = Field(None, max_length=500)
    tone: str | None = Field(None, max_length=50)
    language: str | None = Field(None, max_length=255)
    response_style: str | None = Field(None, max_length=255)
    fallback_message: str | None = Field(None, max_length=500)
    policy_restriction: str | None = Field(None, max_length=255)
    system_prompt: str | None = None

    @field_validator('category', 'tone', 'language', 'response_style', 'policy_restriction')
    @classmethod
    def validate_short_fields(cls, v, info):
        return sanitize_text(v, info.field_name, strict=True)

    @field_validator('target_audience', 'description', 'fallback_message')
    @classmethod
    def validate_medium_fields(cls, v, info):
        return sanitize_text(v, info.field_name, strict=False)

    @field_validator('system_prompt')
    @classmethod
    def validate_system_prompt(cls, v):
        if v is None:
            return None

        v = v.strip()

        if len(v) > 2000:
            raise ValueError("System prompt must not exceed 2000 characters.")

        # Check for prompt injection patterns
        INJECTION_PATTERNS = [
            r'ignore\s+(all\s+)?(previous|prior|above)\s+instructions?',
            r'you\s+are\s+now\s+(?!going)',
            r'disregard\s+(all\s+)?instructions?',
            r'<\s*system\s*>',
            r'\[INST\]',
            r'###\s*instruction',
            r'----+',  # separator tricks
        ]

        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError("System prompt contains potentially unsafe content.")

        return v

class BehaviorStudioRequestSchema(BaseBehaviorStudioSchema):
    user_id: UUID
    chatbot_id: UUID

class BehaviorStudioResponseSchema(BehaviorStudioRequestSchema):
    id: UUID
    created_at: datetime
    updated_at: datetime


class SystemPromptRequestSchema(BaseModel):
    """Extract the text prompt from workspace"""
    user_id: UUID | None = None
    chatbot_id: UUID | None = None
    system_prompt: str | None = None

    @field_validator('system_prompt')
    @classmethod
    def validate_system_prompt(cls, v):
        if v is None:
            return None

        v = v.strip()

        if len(v) > 2000:
            raise ValueError("System prompt must not exceed 2000 characters.")

        INJECTION_PATTERNS = [
            r'ignore\s+(all\s+)?(previous|prior|above)\s+instructions?',
            r'you\s+are\s+now\s+(?!going)',
            r'disregard\s+(all\s+)?instructions?',
            r'<\s*system\s*>',
            r'\[INST\]',
            r'###\s*instruction',
            r'----+',
        ]

        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError("System prompt contains potentially unsafe content.")

        return v

class SystemPromptResponseSchema(SystemPromptRequestSchema):
    created_at: datetime
    

class PromptSuggestionRequestSchema(BaseModel):
    """Extract prompt + suggestions from workspace after 3 secs delay"""
    user_id: UUID | None = None
    chatbot_id: UUID | None = None
    system_prompt: str | None = None
    suggestions: list[str] | None = None

    @field_validator('system_prompt')
    @classmethod
    def validate_system_prompt(cls, v):
        if v is None:
            return None

        v = v.strip()

        if len(v) > 2000:
            raise ValueError("System prompt must not exceed 2000 characters.")

        INJECTION_PATTERNS = [
            r'ignore\s+(all\s+)?(previous|prior|above)\s+instructions?',
            r'you\s+are\s+now\s+(?!going)',
            r'disregard\s+(all\s+)?instructions?',
            r'<\s*system\s*>',
            r'\[INST\]',
            r'###\s*instruction',
            r'----+',
        ]

        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError("System prompt contains potentially unsafe content.")

        return v

    @field_validator('suggestions')
    @classmethod
    def validate_suggestions(cls, v):
        if v is None:
            return None

        if not v:
            raise ValueError("Suggestions list cannot be empty.")

        INJECTION_PATTERNS = [
            r'ignore\s+(all\s+)?(previous|prior|above)\s+instructions?',
            r'you\s+are\s+now\s+(?!going)',
            r'disregard\s+(all\s+)?instructions?',
            r'<\s*system\s*>',
            r'\[INST\]',
            r'###\s*instruction',
        ]

        sanitized = []
        for i, item in enumerate(v):
            if not isinstance(item, str):
                raise ValueError(f"Suggestion at index {i} must be a string.")

            item = item.strip()

            if not item:
                raise ValueError(f"Suggestion at index {i} cannot be empty or whitespace.")

            if len(item) > 500:
                raise ValueError(f"Suggestion at index {i} exceeds 500 characters.")

            for pattern in INJECTION_PATTERNS:
                if re.search(pattern, item, re.IGNORECASE):
                    raise ValueError(f"Suggestion at index {i} contains potentially unsafe content.")

            sanitized.append(item)

        return sanitized

class PromptSuggestionResponseSchema(PromptSuggestionRequestSchema):
    created_at: datetime
    