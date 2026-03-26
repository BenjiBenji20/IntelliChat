from uuid import UUID
import logging
import grpc
import xai_sdk
from api.modules.chat.llm.base_llm import *
from xai_sdk.chat import Chunk, system, user

logger = logging.getLogger(__name__)

class ChatXAI(BaseLLM):
    def __init__(self, model_name, api_key):
        super().__init__(model_name, api_key)
        self.client = xai_sdk.AsyncClient(api_key=api_key)

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
        Chat with xAI (Grok) using the official xai-sdk.
        Returns a generator object.
        """
        knowledge_block = "\n\n".join(
            f"[{i+1}] {chunk}" for i, chunk in enumerate(knowledge)
        ) if knowledge else ""

        # Build system message using xai_sdk.chat helpers
        messages = []

        if system_prompt and knowledge_block:
            messages.append(system(f"{system_prompt}\n\n{knowledge_block}"))
        elif system_prompt:
            messages.append(system(system_prompt))
        elif knowledge_block:
            messages.append(system(knowledge_block))

        messages.append(user(query))

        try:
            # Chunk.content is the text delta for each streamed chunk
            async for chunk in await self.client.chat.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                chunk: Chunk
                if chunk.content:
                    yield chunk.content

        except Exception as e:
            logger.error(f"[ChatXAI] Streaming error for chatbot {chatbot_id}: {e}")
            raise

    async def test_llm(self) -> bool:
        """Test xAI LLM config if alive by hitting a 1 token response."""
        try:
            await self.client.chat.create(
                model=self.model_name,
                messages=[user("hi")],
                max_tokens=1,
            )
            return True

        except grpc.RpcError as e:
            code = e.code()
            if code == grpc.StatusCode.UNAUTHENTICATED:
                raise LLMAuthError()
            elif code == grpc.StatusCode.NOT_FOUND:
                raise LLMModelNotFoundError()
            elif code == grpc.StatusCode.RESOURCE_EXHAUSTED:
                raise LLMRateLimitError()
            else:
                raise LLMConnectionError(str(e))
            