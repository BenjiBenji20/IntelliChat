import json
from datetime import datetime, timezone

from fastapi import HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio

from api.modules.behavior_studio.behavior_studio_schema import *
from api.modules.behavior_studio.behavior_studio_script import prompt_builder
from api.modules.behavior_studio.behavior_studio_repository import ChatbotBehaviorRepository
from api.models.chatbot_behavior import ChatbotBehavior


class BehaviorStudioService:
    def __init__(self, db: AsyncSession = None):
        self.db = db
        self.chatbot_behavior_repo = ChatbotBehaviorRepository(db) if db else None
        
    """
        Method create_behavior_studio: Persist all the none null fields
        Method update_behavior_studio: Partially update the behavior_studio row with none null fields by chatbot_id
        Method create_prompt: Create (no db) final prompt based on input fields + prompt (extracted from prompt workspace)
        Method ai_suggestions_prompt: Generate a list of suggestions (no db) based prompt
        Method improve_prompt: Improve the prompt (no db) extracted from prompt workspace
        Method simplify_prompt: Simplify prompt (no db) and pass to LLM streamer and return back to client
        
        Helper Method: 
            field_transformer: combines fields + prompt into one structures string
    """

    async def create_behavior_studio(
        self, payload: BehaviorStudioRequestSchema
    ) -> BehaviorStudioResponseSchema:
        """
        Orchestrates the creation of a chatbot behavior profile.

        Flow:
        1. Transform non-null behavior fields into a structured raw prompt.
        2. Pass raw prompt to LLM to generate an optimized system prompt.
        3. Persist the behavior fields + optimized system prompt to the DB.
        4. Return the saved behavior profile as a response schema.
        """
        try:
            improved_prompt = await self.get_improved_prompt(payload=payload)
            
            # transform pydantic schema to dict and remove null fields
            payload_dict = payload.model_dump(exclude_unset=True, exclude_none=True)
            
            # update the payload_dict.system_prompt = improved_promt
            payload_dict.update({"system_prompt": improved_prompt})
            
            # store the fields + improved prompt to db
            create: ChatbotBehavior = await self.chatbot_behavior_repo.create(
                **payload_dict
            )
            
            if create is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to save behavior studio configuration."
                )
                
            return BehaviorStudioResponseSchema(
                id=create.id,
                user_id=create.user_id,
                chatbot_id=create.chatbot_id,
                category=create.category,
                target_audience=create.target_audience,
                description=create.description,
                tone=create.tone,
                language=create.language,
                response_style=create.response_style,
                fallback_message=create.fallback_message,
                policy_restriction=create.policy_restriction,
                system_prompt=create.system_prompt,
                created_at=create.created_at,
                updated_at=create.updated_at
            )
            
        except HTTPException:
            raise

        except Exception as e:
            await self.db.rollback()
            raise e
        

    async def update_behavior_studio(
        self, payload: BehaviorStudioRequestSchema
    ) -> BehaviorStudioResponseSchema:
        """
        Update ChatbotBehavior
        Not to update: id, user_id, created_at and original chatbot_id
        """
        try:
            payload_dict = payload.model_dump(exclude_unset=True, exclude_none=True) # only fields client actually sent
            chatbot_id = payload_dict.get("chatbot_id")
            if not chatbot_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Missing chatbot_id in the update payload."
                )
            
            # Strip out protected fields from payload
            protected_fields = {"id", "user_id", "chatbot_id", "created_at"}
            update_data = {k: v for k, v in payload_dict.items() if k not in protected_fields}
            
            if not update_data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No valid fields provided for chatbot behavior update."
                )
                
            update: ChatbotBehavior = await self.chatbot_behavior_repo.patch_chatbot_behavior(
                chatbot_id=chatbot_id, payload=update_data
            )
            
            if update is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Chatbot behavior failed to update."
                )
            
            return BehaviorStudioResponseSchema(
                id=update.id,
                user_id=update.user_id,
                chatbot_id=update.chatbot_id,
                category=update.category,
                target_audience=update.target_audience,
                description=update.description,
                tone=update.tone,
                language=update.language,
                response_style=update.response_style,
                fallback_message=update.fallback_message,
                policy_restriction=update.policy_restriction,
                system_prompt=update.system_prompt,
                created_at=update.created_at,
                updated_at=update.updated_at
            )
            
        except HTTPException:
            raise

        except Exception as e:
            await self.db.rollback()
            raise e
        
        
    async def get_behavior_studio(
        self, chatbot_id: UUID
    ) -> BehaviorStudioResponseSchema:
        """
        Get ChatbotBehavior by chatbot_id
        """
        try:
            from sqlalchemy import select
            stmt = select(ChatbotBehavior).where(ChatbotBehavior.chatbot_id == chatbot_id)
            result = await self.db.execute(stmt)
            behavior: ChatbotBehavior | None = result.scalar_one_or_none()
            
            if behavior is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Chatbot behavior not found."
                )
                
            return BehaviorStudioResponseSchema(
                id=behavior.id,
                user_id=behavior.user_id,
                chatbot_id=behavior.chatbot_id,
                category=behavior.category,
                target_audience=behavior.target_audience,
                description=behavior.description,
                tone=behavior.tone,
                language=behavior.language,
                response_style=behavior.response_style,
                fallback_message=behavior.fallback_message,
                policy_restriction=behavior.policy_restriction,
                system_prompt=behavior.system_prompt,
                created_at=behavior.created_at,
                updated_at=behavior.updated_at
            )
            
        except HTTPException:
            raise

        except Exception as e:
            raise e
        
        
    async def create_prompt(
        self, payload: BehaviorStudioRequestSchema
    ) -> PromptSuggestionResponseSchema:
        """
        Orchestrates the creation of prompt without DB persistence
        
        Flow:
            1. Transform non-null behavior fields into a structured raw prompt.
            2. Pass raw prompt to LLM to generate an optimized system prompt.
            3. Pass the improved prompt to prompt suggestions generator.
            4. Return the improved prompt and prompt suggestions.
        """
        try:
            prompt = payload.system_prompt
            
            payload_dict = payload.model_dump(exclude_unset=True)
            payload_dict.pop("system_prompt")
            field_list = self.fields_transformer(payload_dict)
            
            fields = "".join(field_list)
            
            # convert strings into json array
            raw_suggestions = await self._run_llm(
                prompt_builder.generate_prompt_suggestions, prompt, fields
            )
            
            try:
                suggestions = json.loads(raw_suggestions) if raw_suggestions else []
            except (json.JSONDecodeError, TypeError):
                suggestions = []
            
            return PromptSuggestionResponseSchema(
                    system_prompt=prompt,
                    suggestions=suggestions,
                    created_at=datetime.now(timezone.utc)
                )
            
        except HTTPException:
            raise

        except Exception as e:
            raise e


    async def ai_suggestions_prompt(
        self, payload: PromptSuggestionRequestSchema
    ) -> StreamingResponse:
        """
        Pass the prompt + suggestions to the service (no db) and let AI improve the 
        prompt based on the selected suggestions.
        """
        try:
            prompt = payload.system_prompt.strip()
            suggestions_str = ", ".join(payload.suggestions) if payload.suggestions is not None else ""
            
            if suggestions_str == "":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Prompt cannot be improve without suggestions."
                )
                
            return StreamingResponse(
                prompt_builder.stream_improve_prompt_based_suggestions(prompt, suggestions_str),
                media_type="application/x-ndjson"
            )
            
        except HTTPException:
            raise
        except Exception as e:
            raise e


    async def improve_prompt(
        self, payload: SystemPromptRequestSchema
    ) -> StreamingResponse:
        """
        Improve the prompt 
        Extract the prompt text from workspace and improve it.
        """
        try:
            prompt = payload.system_prompt
            
            return StreamingResponse(
                prompt_builder.stream_improve_prompt_cycle(prompt),
                media_type="application/x-ndjson"
            )
            
        except HTTPException:
            raise
        except Exception as e:
            raise e


    async def simplify_prompt(
        self, payload: SystemPromptRequestSchema
    ) -> StreamingResponse:
        """
        Simplify prompt and pass to LLM streamer and return back to client
        """
        try:
            prompt = payload.system_prompt
            
            return StreamingResponse(
                prompt_builder.stream_simplify_current_prompt(prompt),
                media_type="application/x-ndjson"
            )
            
        except HTTPException:
            raise
        except Exception as e:
            raise e

        
    # ===================================================================================
    # HELPER METHODS
    # ===================================================================================
    def fields_transformer(self, fields: dict) -> list:
        FIELD_LABELS = {
            "category": "Domain",
            "target_audience": "Target Audience",
            "description": "Description",
            "tone": "Tone",
            "language": "Language",
            "response_style": "Response Style",
            "fallback_message": "Fallback Message",
            "policy_restriction": "Policy Restriction",
        }
        return [
            f"{FIELD_LABELS[k]}: {v}"
            for k, v, in fields.items()
                if k in FIELD_LABELS and v
        ] if fields else []

    
    def structure_fields(self, payload: BehaviorStudioRequestSchema) -> str | None:
        """
        Transform fields + prompt to sentence
        """
        payload_dict = payload.model_dump(exclude_unset=True)
        prompt_bldr = None
        system_prompt = payload_dict.get("system_prompt", None)
        prompt_parts = self.fields_transformer(payload_dict)
        if prompt_parts:
            structured_context = "Configuration Fields:\n" + "\n".join(prompt_parts)
            if system_prompt:
                prompt_bldr = f"{structured_context}\n\nAdditional Instructions:\n{system_prompt}"
            else:
                prompt_bldr = structured_context
        else:
            prompt_bldr = system_prompt or None
            
        return prompt_bldr
    
    
    async def get_improved_prompt(self, payload: BehaviorStudioRequestSchema) -> str | None:
        # Transform non-null behavior fields into a structured raw prompt.
        prompt = self.structure_fields(payload)
        
        if prompt is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot create prompt based on fields."
            )
        
        try:
            # let ai improve the fields with prompt for better prompt engineering
            improved_prompt = await self._run_llm(prompt_builder.execute_prompt_cycle, prompt)
            
        except HTTPException:
            raise  
        
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Improving raw prompt failed."
            )
            
        return improved_prompt or None
    
    
    async def _run_llm(self, coro_func, *args, timeout: float = 60.0) -> str | None:
        """
        Runs an asynchronous LLM script function with a timeout.
        Raises 504 if LLM exceeds the timeout threshold.
        """
        try:
            return await asyncio.wait_for(
                coro_func(*args),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="LLM took too long to respond. Try again."
            )
            