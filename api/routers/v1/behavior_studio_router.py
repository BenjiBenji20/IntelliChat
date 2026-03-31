from fastapi import APIRouter, Depends, status, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from api.db.db_session import get_async_db
from api.dependencies.auth import get_current_user
from api.modules.behavior_studio.behavior_studio_service import BehaviorStudioService
from api.modules.behavior_studio.behavior_studio_schema import *
from api.dependencies.rate_limit import rate_limit_by_user

router = APIRouter(
    prefix="/api/behavior-studio",  
    tags=["IntelliChat AI Behavior Studio Routers"]
)

"""
    (POST) create_behavior_studio: Persist all the none null fields
    (PATCH) update_behavior_studio: Partially update the behavior_studio row with none null fields by chatbot_id
    (POST) create_prompt: Create (no db) final prompt based on input fields + prompt (extracted from prompt workspace)
    (POST) ai_suggestions_prompt: Generate a list of suggestions (no db) based prompt
    (POST) improve_prompt: Improve the prompt (no db) extracted from prompt workspace
    (POST) simplify_prompt: Simplify prompt (no db) and pass to LLM streamer and return back to client
"""

@router.post(
    "/create/{to_optimize}", 
    response_model=BehaviorStudioResponseSchema,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(rate_limit_by_user())]
)
async def create_behavior_studio(
    payload: BehaviorStudioRequestSchema,
    to_optimize: bool = True,
    current_user_id: UUID = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Pass all the fields in AI Behavior Studio to the service (with db)
    """
    if not payload.chatbot_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Configure project's chatbot first before accessing AI Behavior Studio."
        )
    
    service = BehaviorStudioService(db)
    return await service.create_behavior_studio(payload=payload, to_optimize=to_optimize)
    

@router.patch(
    "/update", 
    response_model=BehaviorStudioResponseSchema,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(rate_limit_by_user())]
)
async def update_behavior_studio(
    payload: BehaviorStudioRequestSchema,
    current_user_id: UUID = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Patch update the fields in AI Behavior Studio to the service (with db)
    """
    if not payload.chatbot_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Configure project's chatbot first before accessing AI Behavior Studio."
        )
    
    service = BehaviorStudioService(db)
    return await service.update_behavior_studio(payload=payload)


@router.post(
    "/save/prompt", 
    response_model=PromptSuggestionResponseSchema,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(rate_limit_by_user())]
)
async def create_prompt(
    payload: BehaviorStudioRequestSchema,
    current_user_id: UUID = Depends(get_current_user)
):
    """
    Pass the prompt to the service (no db) and let AI generate suggestions
    based on the input/select fields 
    """
    if not payload.chatbot_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Configure project's chatbot first before accessing AI Behavior Studio."
        )
    
    service = BehaviorStudioService()
    return await service.create_prompt(payload=payload)


@router.post(
    "/generate/prompt", 
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(rate_limit_by_user())]
)
async def generate_prompt(
    payload: BehaviorStudioRequestSchema,
    current_user_id: UUID = Depends(get_current_user)
):
    """
    Pass the prompt to the service (no db) and let AI generate suggestions
    based on the input/select fields 
    """
    if not payload.chatbot_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Configure project's chatbot first before accessing AI Behavior Studio."
        )
    
    service = BehaviorStudioService()
    return await service.generate_prompt(payload=payload)


@router.post(
    "/prompt/ai-suggestion", 
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(rate_limit_by_user())]
)
async def ai_suggestions_prompt(
    payload: PromptSuggestionRequestSchema,
    _: UUID = Depends(get_current_user),
):
    """
    Pass the prompt + suggestions to the service (no db) and let AI improve the 
    prompt based on the selected suggestions 
    """
    if not payload.chatbot_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Configure project's chatbot first before accessing AI Behavior Studio."
        )
        
    service = BehaviorStudioService()
    return await service.ai_suggestions_prompt(payload=payload)


@router.post(
    "/prompt/improve", 
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(rate_limit_by_user())]
)
async def improve_prompt(
    payload: SystemPromptRequestSchema,
    _: UUID = Depends(get_current_user),
):
    """
    Pass the prompt to the service (no db) and let AI rephrase the prompt
    """
    if not payload.chatbot_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Configure project's chatbot first before accessing AI Behavior Studio."
        )
        
    service = BehaviorStudioService()
    return await service.improve_prompt(payload=payload)
        

@router.post(
    "/prompt/simplify", 
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(rate_limit_by_user())]
)
async def simplify_prompt(
    payload: SystemPromptRequestSchema,
    _: UUID = Depends(get_current_user),
):
    """
    Pass the prompt to the service (no db) and let AI simplify the prompt
    """
    if not payload.chatbot_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Configure project's chatbot first before accessing AI Behavior Studio."
        )
        
    service = BehaviorStudioService()
    return await service.simplify_prompt(payload=payload)


@router.get(
    "/{chatbot_id}", 
    response_model=BehaviorStudioResponseSchema,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(rate_limit_by_user())]
)
async def get_behavior_studio(
    chatbot_id: UUID,
    _: UUID = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Get the behavior of the chatbot by chatbot_id
    """
    if not chatbot_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Configure project's chatbot first before accessing AI Behavior Studio."
        )
    
    service = BehaviorStudioService(db)
    return await service.get_behavior_studio(chatbot_id=chatbot_id)
