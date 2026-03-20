import logging
import uuid
from datetime import datetime, timezone
from uuid import UUID
 
from fastapi import HTTPException, status
 
from api.modules.chat.llm.base_llm import BaseLLM
from api.modules.chat.chat_schema import *
from api.modules.retrievals.retrieval_schema import RetrievalResponseSchema
from api.modules.retrievals.retrieval_service import RetrieveEmbeddingsService
 
logger = logging.getLogger(__name__)
 
 
class IntelliChat:
    """
    Pure orchestrator. Knows nothing about providers.
    Receives already-resolved BaseLLM and an optional retrieval service.
    """
 
    def __init__(
        self,
        llm: BaseLLM,
        llm_provider: str,
        retrieval_service: RetrieveEmbeddingsService | None = None,
    ) -> None:
        self.llm = llm
        self.llm_provider = llm_provider
        self.retrieval_service = retrieval_service
 
    async def run(
        self,
        chatbot_id: UUID,
        session_id: str,
        query: str,
        system_prompt: str | None = None,
        temperature: float = 0.70,
        # retrieval params — only used when retrieval_service is present
        embedding_provider: str | None = None,
        embedding_api_key: str | None = None,
        embedding_model_name: str | None = None,
        top_k: int = 5,
    ) -> IntellichatResponseSchema:
 
        retrieval: RetrievalResponseSchema | None = None
        knowledge: list[str] = []
 
        # --- Retrieve knowledge (optional) ---
        if self.retrieval_service:
            if not all([embedding_provider, embedding_api_key, embedding_model_name]):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="Retrieval service is configured but embedding credentials are missing.",
                )
            try:
                from api.modules.retrievals.retrieval_schema import RetrievalRequestSchema
                retrieval: RetrievalResponseSchema = await self.retrieval_service.retrieve_embeddings(
                    chatbot_id=chatbot_id,
                    provider=embedding_provider,
                    api_key=embedding_api_key,
                    model_name=embedding_model_name,
                    payload=RetrievalRequestSchema(query=query, top_k=top_k),
                )
                if retrieval and retrieval.results:
                    knowledge = [chunk.page_content for chunk in retrieval.results]
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"[IntelliChat] Retrieval failed for chatbot {chatbot_id}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Knowledge retrieval failed. Please try again.",
                )
 
        # --- Stream LLM response ---
        full_content = ""
        prompt_tokens = 0
        completion_tokens = 0
 
        try:
            async for chunk in self.llm.chat_ai(
                chatbot_id=chatbot_id,
                query=query,
                knowledge=knowledge,
                temperature=temperature,
                system_prompt=system_prompt,
            ):
                full_content += chunk
 
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[IntelliChat] LLM error for chatbot {chatbot_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="The AI model failed to respond. Please try again.",
            )
 
        # --- Build response schema ---
        now = datetime.now(timezone.utc)
        sources = retrieval.results if retrieval and retrieval.results else []
        
        metadata = ModelMetadataResponse(
            llm_model=self.llm.model_name,
            llm_provider=self.llm_provider,
            temperature=temperature,
            embedding_model_name=embedding_model_name,
            embedding_model_provider=embedding_provider,
            top_k=top_k if self.retrieval_service else None,
        )
 
        return IntellichatResponseSchema(
            id=uuid.uuid4(), 
            session_id=session_id,
            chatbot_id=chatbot_id,
            client=UserResponse(query=query, created_at=now),
            assistant=AssistantResponse(content=full_content, created_at=now),
            sources=sources,
            usage=UsageResponse(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            model_metadata=metadata,
        )
        