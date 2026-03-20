from abc import ABC, abstractmethod
from uuid import UUID
 
 
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
    