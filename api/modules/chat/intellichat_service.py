import asyncio
import logging
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from qdrant_client import AsyncQdrantClient

from api.modules.chat.intellichat import IntelliChat
from api.modules.chat.llm.llm_factory import LLMFactory
from api.modules.chat.chat_schema import IntellichatResponseSchema
from api.modules.retrievals.retrieval_service import RetrieveEmbeddingsService
from api.modules.llm_api_keys.llm_key_repository import LlmKeyRepository
from api.modules.embedding_model_api_keys.embedding_model_key_repository import EmbeddingModelKeyRepository
from api.modules.behavior_studio.behavior_studio_repository import ChatbotBehaviorRepository
from api.configs.settings import settings
from shared.keys import decrypt_secret
from api.modules.cache.redis_service import redis_service, API_KEY_CACHE_PREFIX, API_KEY_CACHE_TTL

logger = logging.getLogger(__name__)

_NONE_SENTINEL = "__none__"  # marks optional fields that are genuinely absent

class IntelliChatService:

    def __init__(self, db: AsyncSession, qdrant: AsyncQdrantClient) -> None:
        self.db = db
        self.qdrant = qdrant
        self.llm_key_repo = LlmKeyRepository(db)
        self.embedding_model_repo = EmbeddingModelKeyRepository(db)
        self.chatbot_behavior_repo = ChatbotBehaviorRepository(db)
        self.cache_prefix = f"{API_KEY_CACHE_PREFIX}(chatbot_config_data)"

    # -------------------------------------------------------------------------
    # chat() — full RAG, all config required
    # -------------------------------------------------------------------------
    async def chat(
        self,
        chatbot_id: UUID,
        session_id: str,
        query: str,
        top_k: int = 5,
    ) -> IntellichatResponseSchema:
        try:
            llm_data, embedding_model_data, system_prompt = await self._get_chatbot_config_data(
                chatbot_id, require_embedding=True
            )

            logger.info(f"[INFO] chatbot {chatbot_id}: all config ready. Starting chat.")

            try:
                llm = LLMFactory.create_llm(
                    model_name=llm_data["llm_name"],
                    api_key=llm_data["llm_api_key"],
                    provider=llm_data["llm_provider"],
                )
            except ValueError as e:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

            orchestrator = IntelliChat(
                llm=llm,
                llm_provider=llm_data["llm_provider"],
                retrieval_service=RetrieveEmbeddingsService(qdrant=self.qdrant, db=self.db),
            )

            return await orchestrator.run(
                chatbot_id=chatbot_id,
                session_id=session_id,
                query=query,
                system_prompt=system_prompt,
                temperature=float(llm_data["temperature"]),
                embedding_provider=embedding_model_data["embedding_provider"],
                embedding_api_key=embedding_model_data["embedding_api_key"],
                embedding_model_name=embedding_model_data["embedding_model_name"],
                top_k=top_k,
            )

        except HTTPException:
            logger.error(f"[ERROR] HTTPException in chat() for chatbot {chatbot_id}.")
            raise
        except Exception as e:
            logger.error(f"[ERROR] Unexpected error in chat() for chatbot {chatbot_id}. Info: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred. Please try again.",
            )


    # -------------------------------------------------------------------------
    # test_chat() — LLM required, retriever optional
    # -------------------------------------------------------------------------
    async def test_chat(
        self,
        chatbot_id: UUID,
        session_id: str,
        query: str,
        top_k: int = 5,
    ) -> IntellichatResponseSchema:
        try:
            llm_data, embedding_model_data, system_prompt = await self._get_chatbot_config_data(
                chatbot_id, require_embedding=False
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

            try:
                llm = LLMFactory.create_llm(
                    model_name=llm_data["llm_name"],
                    api_key=llm_data["llm_api_key"],
                    provider=llm_data["llm_provider"],
                )
            except ValueError as e:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

            orchestrator = IntelliChat(
                llm=llm,
                llm_provider=llm_data["llm_provider"],
                retrieval_service=(
                    RetrieveEmbeddingsService(qdrant=self.qdrant, db=self.db)
                    if has_embedding else None
                ),
            )

            return await orchestrator.run(
                chatbot_id=chatbot_id,
                session_id=session_id,
                query=query,
                system_prompt=system_prompt,
                temperature=float(llm_data.get("temperature", 0.70)) if llm_data else 0.70,
                embedding_provider=embedding_model_data.get("embedding_provider") if embedding_model_data else None,
                embedding_api_key=embedding_model_data.get("embedding_api_key") if embedding_model_data else None,
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
    async def _get_chatbot_config_data(
        self,
        chatbot_id: UUID,
        require_embedding: bool,
    ) -> tuple[dict, dict | None, str | None]:
        """
        Single entry point for all chatbot config reads.

        Flow:
            1. Try Redis Hash  →  unpack and return on hit
            2. On miss, fetch from DB (3 queries)
            3. Fire-and-forget write to Redis Hash
            4. Return fresh data

        What is cached (Redis Hash key: "api_key_(chatbot_config_data):{chatbot_id}"):
            llm_api_key, llm_model_name, llm_provider
            embedding_api_key, embedding_model_name,
            embedding_provider, temperature
            system_prompt
        Optional fields use _NONE_SENTINEL ("__none__") so a missing value
        is distinguishable from a cache miss.

        TTL: 12 hours. Invalidate with invalidate_chatbot_config_data_cache().
        """
        cached = await redis_service.get_hash(key=str(chatbot_id), prefix=self.cache_prefix)

        if cached:
            logger.info(f"[CACHE HIT] api_key_(chatbot_config_data) for chatbot {chatbot_id}.")
            return self._unpack_config_cache(cached, require_embedding, chatbot_id)

        logger.info(f"[CACHE MISS] api_key_(chatbot_config_data) for chatbot {chatbot_id}. Fetching from DB.")

        llm_data = await self.get_llm_data(chatbot_id)
        embedding_model_data = await self.get_embedding_model_data(
            chatbot_id, is_regular_chat=require_embedding
        )
        system_prompt = await self.get_system_prompt(
            chatbot_id, is_regular_chat=require_embedding
        )

        # Fire-and-forget — cache write must not delay the response
        asyncio.create_task(
            self._store_config_cache(chatbot_id, llm_data, embedding_model_data, system_prompt)
        )

        return llm_data, embedding_model_data, system_prompt


    async def _store_config_cache(
        self,
        chatbot_id: UUID,
        llm_data: dict,
        embedding_model_data: dict | None,
        system_prompt: str | None,
    ) -> None:
        """
        Packs all config into one Redis Hash and stores it.

        Stored fields:
            llm_api_key           — decrypted LLM key
            llm_model_name        — e.g. "llama-3.3-70b-versatile"
            llm_provider          — e.g. "Groq"
            embedding_api_key     — decrypted embedding key  | "__none__" if absent
            embedding_model_name  — e.g. "text-embedding-004"| "__none__" if absent
            embedding_provider    — e.g. "google ai studio"  | "__none__" if absent
            temperature           — float as string, default "0.70" if absent
            system_prompt         — raw prompt string        | "__none__" if absent
        """
        payload: dict[str, str] = {
            "llm_api_key":           llm_data["llm_api_key"],
            "llm_name":        llm_data["llm_name"],
            "llm_provider":          llm_data["llm_provider"],
            "temperature":           str(llm_data["temperature"])                 if llm_data else "0.70",
            "embedding_api_key":     embedding_model_data["embedding_api_key"]    if embedding_model_data else _NONE_SENTINEL,
            "embedding_model_name":  embedding_model_data["embedding_model_name"] if embedding_model_data else _NONE_SENTINEL,
            "embedding_provider":embedding_model_data["embedding_provider"]       if embedding_model_data else _NONE_SENTINEL,
            "system_prompt":         system_prompt                                if system_prompt        else _NONE_SENTINEL,
        }

        success = await redis_service.set_hash(
            key=str(chatbot_id),
            data=payload,
            prefix=self.cache_prefix,
            ttl=API_KEY_CACHE_TTL,
        )

        if success:
            logger.info(f"[CACHE SET] api_key_(chatbot_config_data) stored for chatbot {chatbot_id}.")
        else:
            logger.warning(f"[CACHE SET FAILED] api_key_(chatbot_config_data) for chatbot {chatbot_id}.")


    def _unpack_config_cache(
        self,
        cached: dict,
        require_embedding: bool,
        chatbot_id: UUID,
    ) -> tuple[dict, dict | None, str | None]:
        """
        Rebuilds the same dict shapes the DB helpers return,
        so callers are agnostic about whether data came from cache or DB.
        """
        llm_data = {
            "llm_api_key": cached["llm_api_key"],
            "llm_name": cached["llm_name"],
            "llm_provider": cached["llm_provider"],
            "temperature": float(cached["temperature"])
        }

        embedding_absent = cached.get("embedding_api_key") == _NONE_SENTINEL

        if embedding_absent and require_embedding:
            # Config was cached when embedding wasn't set up yet.
            # Raise the same 404 the DB helper would have raised.
            logger.error(f"[ERROR] Cached config missing embedding data for chatbot {chatbot_id}.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No Embedding Model API key. Configure your chatbot first.",
            )

        embedding_model_data: dict | None = None
        if not embedding_absent:
            embedding_model_data = {
                "embedding_api_key": cached["embedding_api_key"],
                "embedding_model_name": cached["embedding_model_name"],
                "embedding_provider": cached["embedding_provider"]
            }

        system_prompt_raw = cached.get("system_prompt")
        system_prompt = None if system_prompt_raw == _NONE_SENTINEL else system_prompt_raw

        return llm_data, embedding_model_data, system_prompt


    # -------------------------------------------------------------------------
    # DB helpers
    # -------------------------------------------------------------------------
    async def get_llm_data(self, chatbot_id: UUID) -> dict:
        try:
            llm_details = await self.llm_key_repo.get_llm_details(chatbot_id)

            if not llm_details:
                logger.error(f"[ERROR] No LLM key found for chatbot {chatbot_id}.")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No LLM API key. Configure your chatbot first.",
                )

            llm_api_key = decrypt_secret(
                encrypted_key=llm_details["api_key_encrypted"],
                encryption_key=settings.ENCRYPTION_KEY,
            )
            return {
                "llm_api_key": llm_api_key,
                "llm_name": llm_details["llm_name"],
                "llm_provider": llm_details["llm_provider"],
                "temperature": float(llm_details["temperature"]),
                
            }

        except HTTPException:
            raise
        except Exception:
            logger.error(f"[ERROR] Failed to fetch LLM key for chatbot {chatbot_id}.")
            raise


    async def get_embedding_model_data(
        self, chatbot_id: UUID, is_regular_chat: bool = False
    ) -> dict | None:
        try:
            details = await self.embedding_model_repo.get_embedding_model_details(chatbot_id)

            if not details and is_regular_chat:
                logger.error(f"[ERROR] No embedding model key found for chatbot {chatbot_id}.")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No Embedding Model API key. Configure your chatbot first.",
                )

            if not details:
                logger.warning(f"[WARN] No embedding model key for chatbot {chatbot_id}. Skipping retrieval.")
                return None

            embedding_api_key = decrypt_secret(
                encrypted_key=details["api_key_encrypted"],
                encryption_key=settings.ENCRYPTION_KEY,
            )
            return {
                "embedding_api_key": embedding_api_key,
                "embedding_model_name": details["embedding_model_name"],
                "embedding_provider": details["embedding_provider"],
            }

        except HTTPException:
            raise
        except Exception:
            logger.error(f"[ERROR] Failed to fetch embedding model key for chatbot {chatbot_id}.")
            raise


    async def get_system_prompt(
        self, chatbot_id: UUID, is_regular_chat: bool = False
    ) -> str | None:
        try:
            system_prompt = await self.chatbot_behavior_repo.get_system_prompt(chatbot_id)

            if not system_prompt and is_regular_chat:
                logger.error(f"[ERROR] No system prompt found for chatbot {chatbot_id}.")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="System prompt not found. Configure your chatbot first.",
                )

            if not system_prompt:
                logger.warning(f"[WARN] No system prompt for chatbot {chatbot_id}. Proceeding without it.")
                return None

            return system_prompt

        except HTTPException:
            raise
        except Exception:
            logger.error(f"[ERROR] Failed to fetch system prompt for chatbot {chatbot_id}.")
            raise
        