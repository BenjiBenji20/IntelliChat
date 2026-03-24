import asyncio
import json
import logging
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from qdrant_client import AsyncQdrantClient

from api.modules.chat.llm.intellichat import IntelliChat
from api.modules.chat.llm.llm_factory import LLMFactory
from api.modules.chat.chat_schema import IntellichatResponseSchema
from api.modules.retrievals.retrieval_service import RetrieveEmbeddingsService
from api.modules.llm_api_keys.llm_key_repository import LlmKeyRepository
from api.modules.embedding_model_api_keys.embedding_model_key_repository import EmbeddingModelKeyRepository
from api.modules.behavior_studio.behavior_studio_repository import ChatbotBehaviorRepository
from api.modules.chatbot.chatbot_repository import ChatbotRepository
from shared.keys import decrypt_key
from api.configs.settings import settings
from api.modules.cache.redis_service import FREQ_CACHE_TTL, redis_service, FREQ_CACHE_PREFIX
from api.modules.chat import query_guardrail as gr

logger = logging.getLogger(__name__)

_NONE_SENTINEL = "__none__"  # marks optional fields that are genuinely absent

class TestIntelliChatService:

    def __init__(self, db: AsyncSession, qdrant: AsyncQdrantClient) -> None:
        self.db = db
        self.qdrant = qdrant
        self.llm_key_repo = LlmKeyRepository(db)
        self.embedding_model_repo = EmbeddingModelKeyRepository(db)
        self.chatbot_behavior_repo = ChatbotBehaviorRepository(db)
        self.chatbot_repo = ChatbotRepository(db)
        self.cache_prefix = f"{FREQ_CACHE_PREFIX}(chatbot_current_state)"

    # -----------------------------------------------------------------------------------------
    # test_chat() — (Chatbot data, LLM required) required, (retriever, system_prompt) optional
    # -----------------------------------------------------------------------------------------
    async def test_chat(
        self,
        project_id: UUID,
        chatbot_id: UUID,
        session_id: str,
        query: str,
        top_k: int = 5,
    ) -> IntellichatResponseSchema:
        try:
            llm_data, chatbot_data, embedding_model_data, system_prompt = await self._get_chatbot_current_state_data(
                project_id=project_id
            )

            # Guard: partial embedding config is ambiguous, reject it
            has_embedding = embedding_model_data is not None
            if has_embedding and not all([
                embedding_model_data.get("embedding_api_key"),
                embedding_model_data.get("embedding_model_name"),
                embedding_model_data.get("embedding_provider"),
            ]):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail=(
                        "Incomplete embedding configuration. "
                        "Ensure embedding API key, model name, and provider are all set."
                    ),
                )

            logger.info(
                f"[INFO] chatbot {chatbot_id}: config ready. "
                f"Retrieval {'enabled' if has_embedding else 'disabled'}. Starting test chat."
            )
            
            raw_llm_api_key = decrypt_key(encrypted_key=llm_data["llm_api_key"], encryption_key=settings.ENCRYPTION_KEY)
            llm_data.pop("llm_api_key")

            try:
                llm = LLMFactory.create_llm(
                    model_name=llm_data["llm_name"],
                    api_key=raw_llm_api_key,
                    provider=llm_data["llm_provider"],
                )
            except ValueError as e:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
            
            # determine if query is greetings to skip semantic search
            is_greeting = gr.query_guardrail.is_greeting(query)

            retrieval_svc = RetrieveEmbeddingsService(qdrant=self.qdrant, db=self.db)
            has_knowledge = False
            if has_embedding and not is_greeting: # bypass get_collection_stats if is-greeting=True
                stats = await retrieval_svc.get_collection_stats(chatbot_id)
                has_knowledge = stats is not None
                
            orchestrator = IntelliChat(
                llm=llm,
                llm_provider=llm_data["llm_provider"],
                retrieval_service=(
                    retrieval_svc if has_embedding and has_knowledge and not is_greeting else None
                ),
            )
            
            raw_embedding_api_key = None
            if has_embedding and embedding_model_data.get("embedding_api_key"):
                raw_embedding_api_key = decrypt_key(
                    encrypted_key=embedding_model_data.get("embedding_api_key"), encryption_key=settings.ENCRYPTION_KEY
                )
                embedding_model_data.pop("embedding_api_key")

            return await orchestrator.run(
                chatbot_id=chatbot_id,
                session_id=session_id,
                query=query,
                system_prompt=system_prompt,
                temperature=float(llm_data.get("temperature", 0.70)) if llm_data else 0.70,
                embedding_provider=embedding_model_data.get("embedding_provider") if embedding_model_data else None,
                embedding_api_key=raw_embedding_api_key,
                embedding_model_name=embedding_model_data.get("embedding_model_name") if embedding_model_data else None,
                top_k=top_k,
            )

        except HTTPException:
            logger.error(f"[ERROR] HTTPException in test_chat() for chatbot {chatbot_id}.")
            raise
        except Exception as e:
            logger.error(f"[ERROR] Unexpected error in test_chat() for chatbot {chatbot_id}. Info: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred. Please try again.",
            )


    # -------------------------------------------------------------------------
    # Cache orchestration
    # -------------------------------------------------------------------------
    async def _get_chatbot_current_state_data(
        self, project_id: UUID,
    ) -> tuple[dict, dict, dict | None, str | None]:
        """
        Single entry point for all chatbot config reads.

        Flow:
            1. Try Redis Hash  →  unpack and return on hit
            2. On miss, fetch from DB (3 queries)
            3. Fire-and-forget write to Redis Hash
            4. Return fresh data

        What is cached (Redis Hash key: "freq_data_(chatbot_current_state):{project_id}"):
            chatbot_data:
                id, application_name, has_memory
            llm_data:
                id, api_key_encrypted, llm_name, provider
            embedding_data:
                id, api_key_encrypted, embedding_model_name, provider
            system_prompt
        Optional fields use _NONE_SENTINEL ("__none__") so a missing value
        is distinguishable from a cache miss.
        """
        cached = await redis_service.get_hash(key=str(project_id), prefix=self.cache_prefix)

        if cached:
            logger.info(f"[CACHE HIT] freq_data_(chatbot_current_state) for chatbot {project_id}.")
            return self._unpack_config_cache(cached)
        
        # Cache miss - fetch from DB
        logger.info(f"[CACHE MISS] freq_data_(chatbot_current_state) for project {project_id}.")
        state = await self.chatbot_repo.get_chatbot_setup_status(project_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Project not found.")

        # Fire and forget cache write
        asyncio.create_task(
            redis_service.set_nested_dict_hash(key=str(project_id), prefix=self.cache_prefix, data=state, ttl=FREQ_CACHE_TTL)
        )

        return cached.get("llm_data"), cached.get("chatbot_data"), cached.get("embedding_data"), cached.get("system_prompt")


    def _unpack_config_cache(
        self, cached: dict,
    ) -> tuple[dict, dict | None, str | None]:
        """
        Rebuilds the same dict shapes the DB helpers return,
        so callers are agnostic about whether data came from cache or DB.
        """
        chatbot_raw = cached.get("chatbot_data")
        llm_raw = cached.get("llm_data")
        embedding_raw = cached.get("embedding_data")
        system_prompt_raw = cached.get("system_prompt")

        if chatbot_raw in (None, _NONE_SENTINEL):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Error chat: Chatbot identity not set yet."
            )

        chatbot_data = json.loads(chatbot_raw)
        llm_data_parsed = json.loads(llm_raw)

        llm_data = {
            "llm_api_key": llm_data_parsed["api_key_encrypted"],
            "llm_name": llm_data_parsed["llm_name"],
            "llm_provider": llm_data_parsed["provider"],
            "temperature": float(llm_data_parsed["temperature"])
        }

        embedding_model_data = None
        if embedding_raw not in (None, _NONE_SENTINEL):
            embedding_parsed = json.loads(embedding_raw)
            embedding_model_data = {
                "embedding_api_key": embedding_parsed["api_key_encrypted"],
                "embedding_model_name": embedding_parsed["embedding_model_name"],
                "embedding_provider": embedding_parsed["provider"]
            }

        system_prompt = None if system_prompt_raw in (None, _NONE_SENTINEL) else system_prompt_raw

        return llm_data, chatbot_data, embedding_model_data, system_prompt
