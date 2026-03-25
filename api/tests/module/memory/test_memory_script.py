import asyncio
import os
from uuid import uuid4
import sys

# Add project root to path to allow absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from api.modules.chat.memory.chat_memory import ChatMemory
from sqlalchemy.ext.asyncio import AsyncSession
from unittest.mock import MagicMock, AsyncMock

# Dummy test LLM to simulate response
class DummyLLM:
    async def chat_ai(self, chatbot_id, query, knowledge, temperature, system_prompt):
        yield "This "
        yield "is "
        yield "a "
        yield "dummy "
        yield "summary."

async def run_test():
    # 1. Provide these UUIDs or they will be generated randomly to test cache directly.
    #    Make sure your UPSTASH_REDIS_URL/TOKEN are globally set or configured in settings.
    chatbot_id_str = "0a3e58d9-debe-4e1e-a062-038d6fabc98d" # Replace with valid UUID if you want
    try:
        from uuid import UUID
        chatbot_id = UUID(chatbot_id_str)
    except:
        chatbot_id = uuid4()
    
    session_id = "test-session-12345"

    # Mock DB session since we only want to test Redis/Logic quickly
    mock_db = MagicMock(spec=AsyncSession)
    chat_memory = ChatMemory(db=mock_db)

    print("--- [TEST] Checking initial memory ---")
    memory_result = await chat_memory.my_memory(chatbot_id, session_id)
    print(f"Initial Memory: {memory_result}\n")

    print("--- [TEST] Adding 11 turns to trigger summarization ---")
    turns = []
    for i in range(11):
        turns.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"})
    
    dummy_llm = DummyLLM()
    success = await chat_memory.cache_turns(chatbot_id, session_id, turns, dummy_llm)
    print(f"Cache Turns Success: {success}")

    # Wait a bit for the async background task to complete summarization
    await asyncio.sleep(2)

    print("\n--- [TEST] Checking memory after summarization ---")
    memory_result = await chat_memory.my_memory(chatbot_id, session_id)
    print("Turns count:", len(memory_result.get("turns", []) or []))
    print("Turns:", memory_result.get("turns"))
    print("Summary:", memory_result.get("summary"))
    print("\nTest completed.")

if __name__ == "__main__":
    asyncio.run(run_test())
