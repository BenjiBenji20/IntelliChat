from uuid import UUID
import logging

from groq import AsyncGroq, AuthenticationError, NotFoundError, RateLimitError

from api.modules.chat.llm.base_llm import *

logger = logging.getLogger(__name__)

class ChatGroq(BaseLLM):
    """
    Supported groq models:
        "Groq": [
            "openai/gpt-oss-120b",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768"
        ]
    """
    
    def __init__(self, model_name, api_key):
        super().__init__(model_name, api_key)
        self.client = AsyncGroq(api_key=api_key)

    
    async def chat_ai(
        self,
        chatbot_id: UUID,
        query: str,
        knowledge: list,
        temperature: float = 0.70,
        system_prompt: str | None = None,
    ): 
        """
        Chat with Groq provider using their official python groq sdk
        Response a generator object.
        """
        knowledge_block = "\n\n".join(
            f"[{i+1}] {chunk}" for i, chunk in enumerate(knowledge)
        ) if knowledge else ""
        
        messages = []
        
        # passes prompts + knowledge        
        if system_prompt:
            full_system = (
                f"{system_prompt}\n\n{knowledge_block}"
                if knowledge_block
                else system_prompt
            )
            messages.append({"role": "system", "content": full_system})
        # passes only knowledge
        elif knowledge_block:
            messages.append({"role": "system", "content": knowledge_block})
 
        messages.append({"role": "user", "content": query})
        
        try:
            stream = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
 
        except Exception as e:
            logger.error(f"[ChatGroq] Streaming error for chatbot {chatbot_id}: {e}")
            raise

    
    async def test_llm(self):
        """Test LLM config if alive by hitting a 1 token to it."""
        try:
            await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
                stream=False
            )
            return True
        
        except AuthenticationError:
            raise LLMAuthError()
        except NotFoundError:
            raise LLMModelNotFoundError()
        except RateLimitError:
            raise LLMRateLimitError()
        except Exception as e:
            raise LLMConnectionError(str(e))
            