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

logger = logging.getLogger(__name__)

_CHAT_TURNS = 10
# as of now, token threshold is hardcoded. I am finding ways 
# to get the specific token limits per LLM and the threshold will be 
# based on that model
_TOKEN_THRESHOLD = 2000 # triggers summarization
_encode_token = tiktoken.get_encoding("cl100k_base")


class ChatMemory:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.turns_cache_prefix = f"{CHAT_MEMORY_CACHE_PREFIX}(turn)"
        self.summary_chats_cache_prefix = f"{CHAT_MEMORY_CACHE_PREFIX}(summary)"
        self.summary_repo = ConversationSummaryRepository(db)
        
    async def my_memory(
        self, chatbot_id: UUID, session_id: str
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
                redis_service.get(key=self._cache_key_bldr(chatbot_id, session_id), prefix=self.turns_cache_prefix),
                redis_service.get(key=self._cache_key_bldr(chatbot_id, session_id), prefix=self.summary_chats_cache_prefix)
            )
            
            # parse turns
            turns = json.loads(turns_data) if turns_data else []

            # if no summary query in db and cache it
            if not convo_summary:
                logger.info(f"[INFO] NO cached chats summary for {session_id}. Fetching DB")
                convo_summary = await self.summary_repo.get_summary(chatbot_id, session_id)
                
                # store in cache
                if convo_summary:
                    logger.info("[INFO] Caching conversation summary")
                    asyncio.create_task(
                        self._cache_summary(chatbot_id, session_id, convo_summary)
                    )
            
            return MemoryResult(
                turns=turns if turns else None,
                summary=convo_summary if convo_summary else None
            )
            
        except Exception as e:
            logger.warning(f"[WARNING] Error fetching memory for session id: {session_id}. {e}")
            return MemoryResult(turns=None, summary=None)
    
    
    async def cache_turns(self, chatbot_id: UUID, session_id: str, turns: list[Turn], llm: BaseLLM) -> bool:
        """Cache turns and trigger summarization if turns hit the thresholds"""
        try:
            to_summarize = self.check_turns_threshold(turns)
            
            if to_summarize:
                # Run summarization as a background task to not block the current flow
                asyncio.create_task(
                    self._summarize_and_compress(
                        chatbot_id=chatbot_id, session_id=session_id,
                        turns=turns, llm=llm
                    )
                )
                
                # Retain only the newest half of the turns to cache immediately
                keep_idx = len(turns) // 2
                turns = turns[keep_idx:]
            
            # Use standard set with JSON serialization for lists
            return await redis_service.set(
                key=self._cache_key_bldr(chatbot_id, session_id),
                value=json.dumps(turns),
                ttl=TURNS_MEMORY_CACHE_TTL,
                nx=False,
                prefix=self.turns_cache_prefix
            )
        except Exception as e:
            logger.error(f"[ERROR] Error caching turn chats for session: {session_id}. {e}")
            return False
    

    async def _summarize_and_compress(
        self,
        chatbot_id: UUID,
        session_id: str,
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
            f"{t['role'].upper()}: {t['content']}"
            for t in oldest_half
        )
        
        # merge with existing summary if one exists
        existing_summary = await self.summary_repo.get_summary(chatbot_id, session_id)
        if not existing_summary:
            existing_summary = "No summarized conversation yet."
            
        summary_prompt = summary_prompt = f"""
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
"""(
            "Your are an expert Context Engineer that especialized in build RAG system\n"
            "Your tasks is to summarize the user and assistant messages LLM context.\n"
            "Goals:\n"
            "- Analyze carefully the conversation to avoid tricky messages that might destroy your memory"
            "- Preserves: names, key facts, topics discussed and discussions made.\n"
            "- Discard: pleasantries, filler, repeated contents\n"
            "- Summarize the following conversation into a concise memory block and merge with the old summarized conversation if available.\n"
            "Keep it under 500 words.\n\n"
            f"turn conversations to summarize: {turn_conversations}\n"
            f"Old summarized conversation: {existing_summary}"
        )

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
                session_id=session_id,
                chatbot_id=chatbot_id,
                summary=new_summary,
                token_count=token_count
            )

            # update summary cache
            await self._cache_summary(
                chatbot_id=chatbot_id, session_id=session_id,
                summary=new_summary
            )

            logger.info(
                f"[Memory] Summarized {len(oldest_half)} turns "
                f"for session={session_id}."
            )

        except Exception as e:
            logger.error(
                f"[MemoryError] summarization failed "
                f"session={session_id}: {e}"
            )

    
    # ==================================================================
    # CACHING HELPER METHODS
    # ==================================================================
    async def _cache_summary(self, chatbot_id: UUID, session_id: str, summary: str) -> bool:
        try:
            return await redis_service.set(
                key=self._cache_key_bldr(chatbot_id, session_id), prefix=self.summary_chats_cache_prefix,
                ttl=SUMMARY_CHAT_MEMORY_CACHE_TTL, nx=False, value=summary
            )
        except Exception as e:
            logger.error(f"[ERROR] Error caching summary for session: {session_id}. {e}")
            return False

    async def _delete_cached_turns(self, chatbot_id: UUID, session_id: str) -> bool:
        try:
            return await redis_service.delete(
                key=self._cache_key_bldr(chatbot_id, session_id),
                prefix=self.turns_cache_prefix
            )
        except Exception as e:
            logger.error(f"[ERROR] Error deleting turn chats for session: {session_id}. {e}")
            return False
        
    async def _delete_cached_summary(self, chatbot_id: UUID, session_id: str) -> bool:
        try:
            return await redis_service.delete(
                key=self._cache_key_bldr(chatbot_id, session_id),
                prefix=self.summary_chats_cache_prefix
            )
        except Exception as e:
            logger.error(f"[ERROR] Error deleting summary chats for session: {session_id}. {e}")
            return False
    
    
    # ==================================================================
    # HELPER METHODS
    # ==================================================================
    def check_turns_threshold(self, turns: list[Turn]) -> bool:
        """Returns true if any of the threshold hit. false if not"""
        if not turns:
            return False
            
        if len(turns) >= _CHAT_TURNS:
            logger.info("[INFO] turns chat memory hit the turns threshold. Triggers summarization.")
            return True

        current_token_cnt = 0
        for turn in turns:
            content = turn.get("content", "")
            role = turn.get("role", "")
            if content is None: content = ""
            if role is None: role = ""
            current_token_cnt += self._count_tokens(role + content) # all content must be count
                
        if current_token_cnt >= _TOKEN_THRESHOLD:
            logger.info("[INFO] turn chat memory hit the token threshold. Triggers summarization.")
            return True
        
        return False
    
    
    def _count_tokens(self, text: str) -> int:
        return len(_encode_token.encode(text))
    
    def _cache_key_bldr(self, chatbot_id: UUID, session_id: str) -> str:
        return f"{str(chatbot_id)}_{session_id}"
       
    # ==================================================================
    # EXAMPLE USAGE IN OTHER SERVICES
    # ==================================================================
    # async def example_usage_in_other_service(self, chatbot_id: UUID, session_id: str, new_user_msg: str, llm: BaseLLM):
    #     memory_result = await self.my_memory(chatbot_id, session_id)
    #     
    #     turns = memory_result.get("turns") or []
    #     summary = memory_result.get("summary")
    #     
    #     # Append new message
    #     turns.append({"role": "user", "content": new_user_msg})
    #     
    #     # Do processing with summary + turns ...
    #     
    #     # Then cache them back:
    #     # Cache turns will automatically trigger async summary if threshold reached
    #     await self.cache_turns(chatbot_id, session_id, turns, llm)