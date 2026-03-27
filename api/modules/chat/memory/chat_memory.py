import asyncio
import json
from fastapi import HTTPException, status
import logging
from uuid import UUID

import tiktoken
from sqlalchemy.ext.asyncio import AsyncSession

from api.cache.redis_service import (
    redis_service, TURNS_MEMORY_CACHE_TTL,
    CHAT_MEMORY_CACHE_PREFIX, SUMMARY_CHAT_MEMORY_CACHE_TTL
)
from api.modules.chat.llm.base_llm import BaseLLM
from api.modules.chat.memory.conversation_summary_repository import ConversationSummaryRepository
from api.modules.chat.memory.memory_schema import MemoryResult, Turn
from shared.ai_models_details import EXPECTED_OUTPUT_TOKENS, FALLBACK_MEMORY_CAP, MAX_IDEAL_CONTEXT_WINDOW_PERCENTAGE

logger = logging.getLogger(__name__)

_encode_token = tiktoken.get_encoding("cl100k_base")


class ChatMemory:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.turns_cache_prefix = f"{CHAT_MEMORY_CACHE_PREFIX}(turn)"
        self.summary_chats_cache_prefix = f"{CHAT_MEMORY_CACHE_PREFIX}(summary)"
        self.summary_repo = ConversationSummaryRepository(db)
        
    async def my_memory(
        self, chatbot_id: UUID, conversation_id: str
    ) -> MemoryResult: 
        """
        Get the current turn turns and add a dict to it.
        Triggers summary if _CHAT_TURNS or _TOKEN_THRESHOLD HIT
        
        Caching format:
        key: {prefix}:{key}
            turns: []
            summary: str
        """
        try:
            turns_data, convo_summary = await asyncio.gather(
                redis_service.get(key=self._cache_key_bldr(chatbot_id, conversation_id), prefix=self.turns_cache_prefix),
                redis_service.get(key=self._cache_key_bldr(chatbot_id, conversation_id), prefix=self.summary_chats_cache_prefix)
            )
            
            # parse turns
            turns = json.loads(turns_data) if turns_data else []

            # if no summary query in db and cache it
            if convo_summary is None:
                logger.info(f"[INFO] NO cached chats summary for {conversation_id}. Fetching DB")
                convo_summary = await self.summary_repo.get_summary(chatbot_id, conversation_id)
                
                # store in cache
                if convo_summary:
                    # Cache real summary
                    logger.info("[INFO] Caching conversation summary")
                    asyncio.create_task(
                        self._cache_summary(chatbot_id, conversation_id, convo_summary)
                    )
                else:
                    # Store the Sentinel Flag to prevent hammering the DB next round
                    logger.info("[INFO] Explicitly caching negative EMPTY flag internally")
                    asyncio.create_task(self._cache_summary(chatbot_id, conversation_id, "EMPTY"))
                    convo_summary = None
                    
            # If Redis returned our exact sentinel flag, instantly skip everything
            elif convo_summary == "EMPTY":
                logger.info("[INFO] Successfully intercepted EMPTY sentinel. Safely skipping DB lookup.")
                convo_summary = None
            
            return MemoryResult(
                turns=turns if turns else None,
                summary=convo_summary if convo_summary else None
            )
            
        except Exception as e:
            logger.warning(f"[WARNING] Error fetching memory for session id: {conversation_id}. {e}")
            return MemoryResult(turns=None, summary=None)
    
    
    async def cache_turns(
        self, chatbot_id: UUID, conversation_id: str, turns: list[Turn], llm: BaseLLM,
        current_query: str, knowledge_list: list[str] = [], system_prompt: str | None = ""
    ) -> bool:
        """Cache turns and trigger summarization if turns hit the thresholds"""
        try:
            context_window = getattr(llm, 'context_window', 8192)
            to_summarize = self.check_turns_threshold(
                turns=turns, llm_context_window=context_window,
                current_query=current_query, knowledge_list=knowledge_list,
                system_prompt=system_prompt
            )
            
            if to_summarize:
                # Run summarization as a background task to not block the current flow
                asyncio.create_task(
                    self._summarize_and_compress(
                        chatbot_id=chatbot_id, conversation_id=conversation_id,
                        turns=turns, llm=llm
                    )
                )
                
                # Retain only the newest half of the turns to cache immediately
                keep_idx = len(turns) // 2
                turns = turns[keep_idx:]
            
            # Use standard set with JSON serialization for lists
            return await redis_service.set(
                key=self._cache_key_bldr(chatbot_id, conversation_id),
                value=json.dumps(turns),
                ttl=TURNS_MEMORY_CACHE_TTL,
                nx=False,
                prefix=self.turns_cache_prefix
            )
        except Exception as e:
            logger.error(f"[ERROR] Error caching turn chats for session: {conversation_id}. {e}")
            return False
    

    async def _summarize_and_compress(
        self,
        chatbot_id: UUID,
        conversation_id: str,
        turns: list[Turn],
        llm: BaseLLM,
    ) -> None:
        """
        Only call this method when thresholds hit to refresh the cache and store 
        new summary.
        
        Compress the oldest half of recent turns into a summary.
        Keeps the newest half in full (handled by caller).
        Stores summary in Database + Redis.
        """
        # old half to summarize
        oldest_half = turns[:len(turns) // 2]
        turn_conversations = "\n".join(
            f"[{t.get('messaged_at', 'N/A')}] {t['role'].upper()}: {t['content']}"
            for t in oldest_half
        )
        
        # merge with existing summary if one exists
        existing_summary = await self.summary_repo.get_summary(chatbot_id, conversation_id)
        if existing_summary is None:
            existing_summary = "No summarized conversation yet."
            
        summary_prompt = f"""
You are a Context Engineer responsible for maintaining a compact, high-quality conversation memory for an LLM system.

TASK:
Update the existing memory using the new conversation turns.

INPUT:
- Existing Memory (may be empty)
- New Conversation Turns

RULES:
1. Preserve only high-value information:
   - user identity, preferences, goals
   - key facts, decisions, constraints
   - ongoing tasks or problems

2. Remove:
   - greetings, filler, repetition
   - redundant or outdated information

3. Merge intelligently:
   - Do NOT duplicate existing facts
   - Update outdated or conflicting info with the latest
   - Keep the most relevant version

4. Compress aggressively:
   - Use short, information-dense phrases
   - Avoid full sentences when possible
   
5. Handle time intelligently:
   - Include timestamps ONLY if they add meaningful contexts
   - Prefer short formats: (Nov 2025), (Mar 2026)
   - Use relative terms when exact date is not important (e.g., "recently", "currently")
   - Avoid repeating timestamps for unchanged facts

OUTPUT FORMAT (STRICT):

User Profile:
- ...

Key Facts:
- ...

Ongoing Context:
- ...

Recent Updates:
- ...

CONSTRAINTS:
- Max ~250 words
- Prefer bullet points over paragraphs
- Be concise but precise

---

Existing Memory:
{existing_summary}

---

New Conversation:
{turn_conversations}
"""
        try:
            # use the same LLM instance — small prompt, fast + cheap
            new_summary = ""
            async for chunk in llm.chat_ai(
                chatbot_id=chatbot_id,
                query=summary_prompt,
                knowledge=[],
                temperature=0.3,    # low temp for factual compression
                system_prompt=None,
            ):
                new_summary += chunk

            if not new_summary:
                return
            
            token_count = self._count_tokens(new_summary)

            # persist to Database
            await self.summary_repo.save_summary(
                conversation_id=conversation_id,
                chatbot_id=chatbot_id,
                summary=new_summary,
                token_count=token_count
            )

            # update summary cache
            await self._cache_summary(
                chatbot_id=chatbot_id, conversation_id=conversation_id,
                summary=new_summary
            )

            logger.info(
                f"[Memory] Summarized {len(oldest_half)} turns "
                f"for session={conversation_id}."
            )

        except Exception as e:
            logger.error(
                f"[MemoryError] summarization failed "
                f"session={conversation_id}: {e}"
            )

    
    # ==================================================================
    # CACHING HELPER METHODS
    # ==================================================================
    async def _cache_summary(self, chatbot_id: UUID, conversation_id: str, summary: str) -> bool:
        try:
            return await redis_service.set(
                key=self._cache_key_bldr(chatbot_id, conversation_id), prefix=self.summary_chats_cache_prefix,
                ttl=SUMMARY_CHAT_MEMORY_CACHE_TTL, nx=False, value=summary
            )
        except Exception as e:
            logger.error(f"[ERROR] Error caching summary for session: {conversation_id}. {e}")
            return False

    async def _delete_cached_turns(self, chatbot_id: UUID, conversation_id: str) -> bool:
        try:
            return await redis_service.delete(
                key=self._cache_key_bldr(chatbot_id, conversation_id),
                prefix=self.turns_cache_prefix
            )
        except Exception as e:
            logger.error(f"[ERROR] Error deleting turn chats for session: {conversation_id}. {e}")
            return False
        
    
    # ==================================================================
    # HELPER METHODS
    # ==================================================================
    def check_turns_threshold(
        self, turns: list[Turn], llm_context_window: int,
        current_query: str, knowledge_list: list[str] = [], 
        system_prompt: str | None = ""
    ) -> bool:
        """Returns true if the token limits hit the dynamic threshold. false if not"""
        if not turns:
            return False
        
        # Calculate static overhead tokens
        overhead_content = (current_query or "") + (system_prompt or "")
        if knowledge_list:
            overhead_content += "".join(knowledge_list)
        
        overhead_tokens = self._count_tokens(overhead_content)

        # Available space strictly for memory (turns + existing summary)
        available_for_memory = llm_context_window - overhead_tokens - EXPECTED_OUTPUT_TOKENS
        
        # Soft cap for turns to prevent bloating if context window is huge
        soft_memory_cap = min(int(llm_context_window * MAX_IDEAL_CONTEXT_WINDOW_PERCENTAGE), FALLBACK_MEMORY_CAP)
        
        # True Budget is the tighter bound of physical space vs. soft cap limits
        memory_budget = min(available_for_memory, soft_memory_cap)
        
        if memory_budget <= 0:
            logger.warning("[WARNING] Context Window EXHAUSTED by Knowledge and System Prompt! Forcing aggressive summarization.")
            return True 
            
        # Count the current turns payload
        turns_content = "".join([(t.get("role","") + t.get("content","") + t.get("messaged_at","")) for t in turns])
        turns_tokens = self._count_tokens(turns_content)
        
        if turns_tokens >= memory_budget:
            logger.info(f"[INFO] turns token count ({turns_tokens}) hit dynamic budget ({memory_budget}). Triggers summarization.")
            return True
        
        return False
    
    
    def reduce_knowledge(
        self, turns: list[Turn], llm_context_window: int,
        current_query: str, knowledge_list: list[str] = [], 
        system_prompt: str | None = ""
    ) -> list[str]:
        """
        Returns knowledge list for LLM
        This avoids bloated knowledge for context injection
        
        if user passes top_k=1000
        that would bloat the LLM and easily consume all llm context window and 
        constantly cache turns and early summary.
        
        to avoid this, cut the knowledge if it hits the knowledge ceiling budget
        against the llm context window. Cut only the knowledge with low score.
        """
        # Calculate static overhead tokens
        overhead_content = (current_query or "") + (system_prompt or "")

        # Count the current turns payload
        turns_content = "".join([(t.get("role","") + t.get("content","") + t.get("messaged_at","")) for t in turns])
        overhead_content += turns_content
        overhead_tokens = self._count_tokens(overhead_content)
        
        RESERVED_OUTPUT_TOKENS = 2000
        soft_memory_cap = min(int(llm_context_window * 0.15), FALLBACK_MEMORY_CAP)
        
        # calculate remaining space for knowledge
        remaining_for_knowledge = (
            llm_context_window 
            - overhead_tokens 
            - RESERVED_OUTPUT_TOKENS 
            - soft_memory_cap
        )

        if remaining_for_knowledge <= 0:
            logger.warning("[WARNING] No space left for knowledge. Returning empty list.")
            return []
        
        selected_knowledge = []
        total_tokens = 0

        for doc in knowledge_list:
            doc_tokens = self._count_tokens(doc)

            if total_tokens + doc_tokens > remaining_for_knowledge:
                break

            selected_knowledge.append(doc)
            total_tokens += doc_tokens

        logger.info(
            f"[INFO] Knowledge reduced: {len(selected_knowledge)}/{len(knowledge_list)} chunks "
            f"({total_tokens}/{remaining_for_knowledge} tokens used)"
        )

        return selected_knowledge
        
    
    def _count_tokens(self, text: str) -> int:
        return len(_encode_token.encode(text))
    
    def _cache_key_bldr(self, chatbot_id: UUID, conversation_id: str) -> str:
        return f"{str(chatbot_id)}_{conversation_id}"
