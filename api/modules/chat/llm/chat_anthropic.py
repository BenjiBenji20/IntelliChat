from uuid import UUID
import logging
from anthropic import (
    AsyncAnthropic,
    AuthenticationError,
    NotFoundError,
    RateLimitError,
    APIConnectionError,
    InternalServerError,
)
from api.modules.chat.llm.base_llm import *

logger = logging.getLogger(__name__)

class ChatAnthropic(BaseLLM):
    def __init__(self, model_name, api_key):
        super().__init__(model_name, api_key)
        self.client = AsyncAnthropic(api_key=api_key)

    async def chat_ai(
        self,
        chatbot_id: UUID,
        query: str,
        knowledge: list,
        temperature: float = 0.70,
        max_tokens: int = 1024,
        system_prompt: str | None = None,
    ):
        """
        Chat with Anthropic provider using their official python anthropic sdk.
        Returns a generator object.
        """
        knowledge_block = "\n\n".join(
            f"[{i+1}] {chunk}" for i, chunk in enumerate(knowledge)
        ) if knowledge else ""

        # Build system string — Anthropic takes system as a separate param
        system = None
        if system_prompt and knowledge_block:
            system = f"{system_prompt}\n\n{knowledge_block}"
        elif system_prompt:
            system = system_prompt
        elif knowledge_block:
            system = knowledge_block

        messages = [{"role": "user", "content": query}]

        try:
            async with self.client.messages.stream(
                model=self.model_name,
                messages=messages,
                system=system,           
                temperature=temperature,
                max_tokens=max_tokens,   
            ) as stream:
                async for text in stream.text_stream:  
                    yield text

        except Exception as e:
            logger.error(f"[ChatAnthropic] Streaming error for chatbot {chatbot_id}: {e}")
            raise

    async def test_llm(self) -> bool:
        """Test LLM config if alive by hitting a 1 token to it."""
        try:
            await self.client.messages.create(  
                model=self.model_name,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,           
            )
            return True

        except AuthenticationError:
            raise LLMAuthError()
        except NotFoundError:
            raise LLMModelNotFoundError()
        except RateLimitError:
            raise LLMRateLimitError()
        except (APIConnectionError, InternalServerError) as e:
            raise LLMConnectionError(str(e))
        