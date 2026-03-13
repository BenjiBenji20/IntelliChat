import asyncio
import os
import json
from uuid import UUID
from unittest.mock import AsyncMock
from datetime import datetime, timezone

import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Setup paths for python imports
from api.modules.behavior_studio.behavior_studio_service import BehaviorStudioService
from api.modules.behavior_studio.behavior_studio_schema import (
    BehaviorStudioRequestSchema,
    PromptSuggestionRequestSchema,
    SystemPromptRequestSchema
)
from fastapi import HTTPException
from pydantic import ValidationError

class MockChatbotBehavior:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.id = UUID("00000000-0000-0000-0000-000000000001")
        if not hasattr(self, 'created_at'):
            self.created_at = datetime.now(timezone.utc)
        if not hasattr(self, 'updated_at'):
            self.updated_at = datetime.now(timezone.utc)

async def run_tests():
    # Setup paths
    results_dir = os.path.join(project_root, 'tests/results/module')
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, 'behavior_studio_service.txt')
    
    # Initialize service with mock DB
    service = BehaviorStudioService()
    service.db = AsyncMock()
    service.chatbot_behavior_repo = AsyncMock()
    
    chatbot_id = UUID("4f7a7786-47f8-47f0-abb8-b7515ee1f6b3")
    user_id = UUID("621e2246-b147-41c4-8562-c96cf9adf5f7")
    
    def mock_create(**kwargs):
        return MockChatbotBehavior(**kwargs)
    
    def mock_update(*args, **kwargs):
        payload = kwargs.get('payload', {})
        if not payload and len(args) > 1:
            payload = args[1]
        c_id = kwargs.get('chatbot_id', chatbot_id)
        if not c_id and len(args) > 0:
            c_id = args[0]
        return MockChatbotBehavior(user_id=user_id, chatbot_id=c_id, **payload)
    
    service.chatbot_behavior_repo.create.side_effect = mock_create
    service.chatbot_behavior_repo.patch_chatbot_behavior.side_effect = mock_update

    with open(results_file, "w", encoding="utf-8") as f:
        f.write("=== Behavior Studio Service Test Results ===\n\n")

    def append_result(method_name, result):
        with open(results_file, "a", encoding="utf-8") as f:
            f.write(f"--- {method_name} ---\n")
            if isinstance(result, str):
                f.write(result + "\n\n")
            elif hasattr(result, 'model_dump_json'):
                f.write(result.model_dump_json(indent=2) + "\n\n")
            elif hasattr(result, 'model_dump'):
                f.write(json.dumps(result.model_dump(), indent=2, default=str) + "\n\n")
            else:
                f.write(str(result) + "\n\n")

    payload_data = {
        "chatbot_id": chatbot_id,
        "user_id": user_id,
        "category": "E-commerce",
        "target_audience": "Students",
        "description": "Help student to choose which school supplies they need",
        "tone": "Friendly, Persuasive",
        "language": "Taglish. Kapag nag ask in Tagalog, respond in Tagalog, else English then respond in English.",
        "response_style": "Conversational",
        "fallback_message": "I'm sorry, I couldn't answer that query, please contact us in messenger instead.",
        "policy_restriction": "Refuse unrelated questions",
        "system_prompt": "You are a friendly and persuasive shopping assistant for an e-commerce store that sells school supplies. Your goal is to help students choose the right school supplies based on their needs, preferences, and school requirements.\n\nYour primary audience is students, so communicate in a way that is approachable, supportive, and easy to understand. Always maintain a friendly and conversational tone when interacting with users.\n\nLanguage rules:\nRespond in Taglish. If the user asks a question in Tagalog, respond in Tagalog. If the user asks in English, respond in English. Match the user's language whenever possible.\n\nBehavior guidelines:\nHelp students identify the school supplies they may need such as notebooks, pens, calculators, art materials, and other study essentials. When appropriate, suggest useful items and explain briefly why they might be helpful for school activities or studying.\n\nResponse style:\nKeep responses conversational and easy to follow. Avoid overly technical explanations and instead guide the user like a helpful store assistant.\n\nPolicy restrictions:\nOnly answer questions related to school supplies, studying tools, and relevant shopping assistance. If a user asks something unrelated, politely refuse and explain that you can only help with school supply recommendations.\n\nFallback behavior:\nIf you do not have enough information to answer the user's question, or if the requested information is unavailable, respond with the following message:\n\"I'm sorry, I couldn't answer that query, please contact us in messenger instead.\"\n\nAlways prioritize helpfulness, clarity, and a positive shopping experience for students looking for school supplies."
    }

    try:
        print("Running fields_transformer...")
        req_schema = BehaviorStudioRequestSchema(**payload_data)
        fields_result = service.fields_transformer(req_schema)
        append_result("fields_transformer", fields_result)

        print("Running create_behavior_studio...")
        create_result = await service.create_behavior_studio(req_schema)
        # Avoid flooding the log if there's too much data, wait we will just print what it returns.
        append_result("create_behavior_studio", {"id": create_result.id, "system_prompt": create_result.system_prompt})
        
        print("Running update_behavior_studio...")
        update_result = await service.update_behavior_studio(req_schema)
        append_result("update_behavior_studio", {"id": update_result.id, "system_prompt": update_result.system_prompt})

        print("Running create_prompt...")
        create_prompt_result = await service.create_prompt(req_schema)
        append_result("create_prompt", {"system_prompt": create_prompt_result.system_prompt, "suggestions": create_prompt_result.suggestions})

        improved_system_prompt = create_result.system_prompt
        
        print("Running ai_suggestions_prompt...")
        suggestions = ["Make tone more playful", "Add refusal fallback for refunds"]
        sugg_schema = PromptSuggestionRequestSchema(
            system_prompt=improved_system_prompt,
            suggestions=suggestions
        )
        ai_suggestions_result = await service.ai_suggestions_prompt(sugg_schema)
        append_result("ai_suggestions_prompt", ai_suggestions_result)

        print("Running improve_prompt...")
        improve_schema = SystemPromptRequestSchema(
            system_prompt=ai_suggestions_result.system_prompt
        )
        improve_result = await service.improve_prompt(improve_schema)
        append_result("improve_prompt", improve_result)

        print("Running simplify_prompt...")
        simplify_schema = SystemPromptRequestSchema(
            system_prompt=improve_result.system_prompt
        )
        simplify_result = await service.simplify_prompt(simplify_schema)
        append_result("simplify_prompt", simplify_result)

        print(f"Tests complete. Results saved to {results_file}")
    
    except Exception as e:
        print(f"Test run failed with error: {e}")
        append_result("ERROR", f"Test execution failed.\nException: {str(e)}")

if __name__ == "__main__":
    asyncio.run(run_tests())
