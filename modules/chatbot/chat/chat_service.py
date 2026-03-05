from datetime import datetime, timezone

from fastapi import HTTPException, status
from groq import Groq
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID
import asyncio

from schemas.chatbot_schema import *
from modules.chatbot.chatbot_repository import ChatbotRepository
from utils.secret_key_utils import *


class ChatService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.chatbot_repo = ChatbotRepository(db)

    async def chat(
        self,
        chat: RequestChat,
        project_id: UUID,
        chatbot_id: UUID,
        environment: str = "development"
    ) -> ResponseChat:
        try:
            message: str = chat.message

            rows = await  self.chatbot_repo.get_models_identities(
                chatbot_id=chatbot_id
            )
            
            # decrypt api keys
            llm_encrypted_key = rows["llm_encrypted_key"]
            embedding_model_encrypted_key = rows["embedding_model_encrypted_key"] # to use
            
            response = None
            
            if llm_encrypted_key:
                try:
                    response = await asyncio.to_thread(
                        self.chat_llm,
                        [{"role": chat.role.lower(), "content": chat.message}],
                        decrypt_secret(llm_encrypted_key),
                        rows["llm_model_name"],
                        float(rows["llm_temperature"]),
                        rows["llm_provider"]
                    )
                except Exception as e:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Intellichat error: {str(e)}" 
                    )
                    
            if response is None:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="LLM not responding."
                )
                
            return ResponseChat(
                role=chat.role.lower(),
                message=response,
                environment=environment,
                created_at=datetime.now(timezone.utc)
            )

        except HTTPException:
            raise
        except Exception as e:
            await self.db.rollback()
            raise e 


    def chat_llm(
        self, 
        request: list[dict], 
        api_key: str, 
        model_name: str,
        temperature: float,
        provider: str,
    ) -> str:
        """
        Test LLM if working
        """
        llm = Groq(api_key=api_key)
        
        test_llm = llm.chat.completions.create(
            model=model_name,
            messages=request,
            temperature=temperature,
            max_tokens=1024,
            stream=True
        )
        
        # concat chunks as they arrive
        response = ""
        for reply in test_llm:
            if reply.choices[0].delta.content:
                response += reply.choices[0].delta.content
        return response
