from uuid import UUID
import logging
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError
from api.modules.chat.llm.base_llm import *

logger = logging.getLogger(__name__)

class ChatGoogle(BaseLLM):
    def __init__(self, model_name, api_key):
        super().__init__(model_name, api_key)
        self.client = genai.Client(api_key=api_key)

    async def chat_ai(
        self,
        chatbot_id: UUID,
        query: str,
        knowledge: list,
        temperature: float = 0.70,
        max_output_tokens: int = 1024,
        system_prompt: str | None = None,
    ):
        """
        Chat with Google provider using their official google-genai SDK.
        Returns a generator object.
        """
        knowledge_block = "\n\n".join(
            f"[{i+1}] {chunk}" for i, chunk in enumerate(knowledge)
        ) if knowledge else ""

        # Build system instruction — Google uses system_instruction in config
        system_instruction = None
        if system_prompt and knowledge_block:
            system_instruction = f"{system_prompt}\n\n{knowledge_block}"
        elif system_prompt:
            system_instruction = system_prompt
        elif knowledge_block:
            system_instruction = knowledge_block

        config = types.GenerateContentConfig(
            system_instruction=system_instruction, 
            temperature=temperature,
            max_output_tokens=max_output_tokens,   
        )

        try:
            async for chunk in await self.client.aio.models.generate_content_stream(
                model=self.model_name,
                contents=query,                     
                config=config,
            ):
                if chunk.text:
                    yield chunk.text                

        except Exception as e:
            logger.error(f"[ChatGoogle] Streaming error for chatbot {chatbot_id}: {e}")
            raise

    async def test_llm(self) -> bool:
        """Test LLM config if alive by hitting a 1 token response."""
        try:
            await self.client.aio.models.generate_content(   
                model=self.model_name,
                contents="hi",
                config=types.GenerateContentConfig(
                    max_output_tokens=1,
                ),
            )
            return True

        except ClientError as e:
            if e.code == 401:
                raise LLMAuthError()
            elif e.code == 404:
                raise LLMModelNotFoundError()
            elif e.code == 429:
                raise LLMRateLimitError()
            else:
                raise LLMConnectionError(str(e))
        except ServerError as e:
            raise LLMConnectionError(str(e))
        