import asyncio
import json
import logging
import typing
import uuid
from datetime import datetime, timezone
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
 
from api.modules.chat.llm.base_llm import BaseLLM
from api.modules.chat.chat_schema import *
from api.modules.retrievals.retrieval_schema import CollectionStatsSchema, RetrievalResponseSchema, RetrievalRequestSchema
from api.modules.retrievals.retrieval_service import RetrieveEmbeddingsService
from api.modules.chat.memory.chat_memory import ChatMemory
from api.cache.redis_service import (
    EMBEDDING_CACHE_PREFIX, 
    EMBEDDING_CACHE_TTL, 
    FREQ_CACHE_PREFIX, 
    redis_service
)
from shared.ai_models_details import (
    EXPECTED_OUTPUT_TOKENS, 
    FALLBACK_MEMORY_CAP, 
    MAX_IDEAL_CONTEXT_WINDOW_PERCENTAGE, 
    THEORETICAL_CHUNK_SIZE
)

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
        has_memory: bool,
        db: AsyncSession,
        retrieval_service: RetrieveEmbeddingsService | None = None,
    ) -> None:
        self.llm = llm
        self.llm_provider = llm_provider
        self.db = db
        self.has_memory = has_memory
        self.retrieval_service = retrieval_service
 
    async def run(
        self,
        chatbot_id: UUID,
        conversation_id: str,
        query: str,
        stream: bool = False,
        system_prompt: str | None = None,
        temperature: float = 0.70,
        # retrieval params — only used when retrieval_service is present
        embedding_provider: str | None = None,
        embedding_api_key: str | None = None,
        embedding_model_name: str | None = None,
        filters: list[RetrievalFilter] | None = None,
        total_docs: int | None = None, # total docs in qdrant under collection name
        top_k: int = 5,
    ) -> typing.Union[IntellichatResponseSchema, typing.AsyncGenerator]:
        generator = self._core_logic_generator(
            chatbot_id=chatbot_id, conversation_id=conversation_id, query=query,
            system_prompt=system_prompt, temperature=temperature,
            embedding_provider=embedding_provider, embedding_api_key=embedding_api_key,
            embedding_model_name=embedding_model_name, filters=filters,
            total_docs=total_docs, top_k=top_k, stream=stream
        )
        if stream:
            return generator
            
        final_schema = None
        async for payload in generator:
            if payload.startswith("data: [DONE]"):
                data_str = payload.replace("data: [DONE] ", "").strip()
                final_schema = IntellichatResponseSchema.model_validate_json(data_str)
        
        if final_schema is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Stream ended without producing a final schema."
            )
        return final_schema

    async def _core_logic_generator(
        self,
        chatbot_id: UUID,
        conversation_id: str,
        query: str,
        stream: bool,
        system_prompt: str | None = None,
        temperature: float = 0.70,
        embedding_provider: str | None = None,
        embedding_api_key: str | None = None,
        embedding_model_name: str | None = None,
        filters: list[RetrievalFilter] | None = None,
        total_docs: int | None = None,
        top_k: int = 5,
    ):
        yield f"data: {json.dumps({'status': 'Initializing...'})}\n\n"
        
        retrieval: RetrievalResponseSchema | None = None
        knowledge: list[str] = []
        actual_top_k = top_k
 
        # --- Start Memory Fetch Concurrently ---
        memory = ChatMemory(self.db)
        memory_task = None
        if self.has_memory:
            memory_task = asyncio.create_task(
                memory.my_memory(chatbot_id=chatbot_id, conversation_id=conversation_id)
            )

        # --- Retrieve knowledge (optional) ---
        if self.retrieval_service:
            yield f"data: {json.dumps({'status': 'Retrieving knowledge...'})}\n\n"
            if not all([embedding_provider, embedding_api_key, embedding_model_name]):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="Retrieval service is configured but embedding credentials are missing.",
                )

            if total_docs is None:
                # get the actual total docs in qdrant and use as ceiling
                stats = await self.retrieval_service.get_collection_stats(chatbot_id)
                if stats:
                    total_docs = stats.total_documents

            # calculating safe max top_k against massive (if eg. top_k=50000) top_k input of user
            # before pinging the qdrant
            soft_memory_cap = min(int(self.llm.context_window * MAX_IDEAL_CONTEXT_WINDOW_PERCENTAGE), FALLBACK_MEMORY_CAP)
            safe_rag_tokens = self.llm.context_window - EXPECTED_OUTPUT_TOKENS - soft_memory_cap - 1000 # reserved 1000 tokenns
            
            math_safe_top_k = max(1, safe_rag_tokens // THEORETICAL_CHUNK_SIZE)
            safe_top_k = min(total_docs, math_safe_top_k) if total_docs is not None else math_safe_top_k
            actual_top_k = min(top_k, safe_top_k)
            
            try:
                retrieval: RetrievalResponseSchema = await self.retrieval_service.retrieve_embeddings(
                    chatbot_id=chatbot_id,
                    provider=embedding_provider,
                    api_key=embedding_api_key,
                    model_name=embedding_model_name,
                    cache_prefix=EMBEDDING_CACHE_PREFIX,
                    cache_ttl=EMBEDDING_CACHE_TTL,
                    payload=RetrievalRequestSchema(
                        query=query, 
                        filters=filters if filters is not None else [],    
                        top_k=actual_top_k
                    ),
                )
                if retrieval and retrieval.results:
                    retrieval.results = sorted(
                        retrieval.results,
                        key=lambda x: x.score,
                        reverse=True # highest score first
                    )
                    
                    knowledge = [chunk.page_content for chunk in retrieval.results]
            except HTTPException:
                raise
            except Exception as e:
                if "404" in str(e) or "Not Found" in str(e):
                    logger.warning(f"[IntelliChat] Vector collection not found for chatbot {chatbot_id}. Proceeding with base LLM knowledge.")
                    knowledge = []
                else:
                    logger.error(f"[IntelliChat] Retrieval failed for chatbot {chatbot_id}: {e}")
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail="Knowledge retrieval failed. Please try again.",
                    )
 
        yield f"data: {json.dumps({'status': 'Thinking...'})}\n\n"

        # --- Stream LLM response ---
        full_contexts = system_prompt
        conversation_summary = None
        turns = []
        
        if self.has_memory and memory_task:
            whole_memory = await memory_task
            turns = whole_memory.get("turns") or []
            conversation_summary = whole_memory.get("summary") or "No summary yet."
            
            formatted_turns = "\n".join([f"[{t.get('messaged_at', 'N/A')}] {t.get('role', 'User').title()}: {t.get('content', '')}" for t in turns])

            full_contexts = f"""
            {system_prompt}
            
            Recent chats:
            {formatted_turns}
            
            Conversation summary:
            {conversation_summary}
            """
 
        # reduce knowledge list to not bloat the llm
        if knowledge:
            knowledge = memory.reduce_knowledge(
                turns=turns, llm_context_window=self.llm.context_window, current_query=query, 
                knowledge_list=knowledge, system_prompt=full_contexts
            )

        yield f"data: {json.dumps({'status': 'Generating response...'})}\n\n"

        full_content = ""
        is_cancelled = False
        try:
            async for chunk in self.llm.chat_ai(
                chatbot_id=chatbot_id,
                query=query,
                knowledge=knowledge,
                temperature=temperature,
                system_prompt=full_contexts,
            ):
                full_content += chunk
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
 
        except asyncio.CancelledError:
            is_cancelled = True
            logger.info(f"[IntelliChat] Client disconnected during generation for chatbot {chatbot_id}")
            raise
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[IntelliChat] LLM error for chatbot {chatbot_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="The AI model failed to respond. Please try again.",
            )
        finally:
            now = datetime.now(timezone.utc)
            current_time_str = datetime.now().strftime('%b %d, %Y %I:%M %p')
                
            if self.has_memory and full_content: 
                turns.append({"role": "user", "content": query, "messaged_at": current_time_str})
                turns.append({"role": "assistant", "content": full_content, "messaged_at": current_time_str})
                    
                asyncio.create_task(
                    memory.cache_turns(
                        chatbot_id=chatbot_id, conversation_id=conversation_id,
                        turns=turns, llm=self.llm, current_query=query, 
                        knowledge_list=knowledge, system_prompt=system_prompt
                    )
                )

        if not is_cancelled:
            # build token receipt
            query_tokens, prompt_tokens, knowledge_tokens, recent_memory_tokens, llm_response_tokens, total_tokens  = memory.token_receipt(
                query=query, system_prompt=system_prompt, knowledge_list=knowledge if knowledge else None,
                recent_memory=turns or [], llm_response=full_content if full_content else None
            )
            convo_summary_tokens = memory._count_tokens(conversation_summary) if conversation_summary else 0
    
            sources = retrieval.results if retrieval and retrieval.results else []
            
            metadata = ModelMetadataResponse(
                llm_model=self.llm.model_name,
                llm_provider=self.llm_provider,
                temperature=temperature,
                embedding_model_name=embedding_model_name,
                embedding_model_provider=embedding_provider,
                top_k=actual_top_k if self.retrieval_service else None,
            )
    
            schema = IntellichatResponseSchema(
                id=uuid.uuid4(), 
                conversation_id=conversation_id,
                chatbot_id=chatbot_id,
                stream=stream,
                client=UserResponse(query=query, created_at=now),
                assistant=AssistantResponse(content=full_content, created_at=now),
                sources=sources,
                usage=UsageResponse(
                    query_tokens=query_tokens,
                    prompt_tokens=prompt_tokens,
                    knowledge_tokens=knowledge_tokens,
                    llm_response_tokens=llm_response_tokens,
                    recent_memory_tokens=recent_memory_tokens,
                    summarized_memory_tokens=convo_summary_tokens,
                    total_tokens=total_tokens + convo_summary_tokens
                ),
                model_metadata=metadata,
            )
            yield f"data: [DONE] {schema.model_dump_json()}\n\n"
        