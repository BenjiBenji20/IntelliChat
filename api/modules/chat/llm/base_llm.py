from abc import ABC, abstractmethod
from uuid import UUID

from fastapi import HTTPException
 
 
class BaseLLM(ABC):
 
    def __init__(self, model_name: str, api_key: str) -> None:
        self.model_name = model_name
        self.api_key = api_key
 
    @abstractmethod
    async def chat_ai(
        self,
        chatbot_id: UUID,
        query: str,
        knowledge: list,
        temperature: float = 0.70,
        system_prompt: str | None = None,
    ):
        """Returns an async generator (streaming response)."""
        pass
    
    
    @abstractmethod
    async def test_llm(self):
        """Testing LLM configuration on LLM key registration"""
        pass
    
    
    @staticmethod
    def raise_http_from_llm_error(e: Exception):
        if isinstance(e, LLMAuthError):
            raise HTTPException(status_code=401, detail="Invalid LLM API key.")
        if isinstance(e, LLMModelNotFoundError):
            raise HTTPException(status_code=404, detail="Model not found on this provider.")
        if isinstance(e, LLMRateLimitError):
            raise HTTPException(status_code=429, detail="LLM API key rate limit exceeded.")
        if isinstance(e, LLMConnectionError):
            raise HTTPException(status_code=502, detail="Could not reach the LLM provider.")
        raise HTTPException(status_code=500, detail="Unexpected LLM error.")

    
# provider agnostic exception hanlers
class LLMAuthError(Exception): pass
class LLMModelNotFoundError(Exception): pass
class LLMRateLimitError(Exception): pass
class LLMConnectionError(Exception): pass
