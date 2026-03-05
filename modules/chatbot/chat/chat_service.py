from fastapi import HTTPException, status
from groq import Groq
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID
import asyncio

from schemas.chatbot_schema import *
from modules.chatbot.chatbot_repository import ChatbotRepository


class ChatService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.chatbot_repo = ChatbotRepository(db)

    async def chat(
        self,
        chat: RequestChat,
        project_id: UUID,
        chatbot_id: UUID
    ):
        try:
            message: str = chat.message

            rows = await  self.chatbot_repo.get_models_identities(
                chatbot_id=chatbot_id
            )

            if rows is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Chatbot model identity not found."
                )
            
            # decrypt api keys
            hashed_llm_key = rows.get("llm_encrypted_key", "")
            hashed_embedding_model_key = rows.get("embedding_model_encrypted_key")
            
            response = await asyncio.to_thread(
                self.chat_llm,
                request=[{"role": chat.role.lower(), "content": message}],
                api_key
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
        
        if test_llm is None:
            return None
        
        # concat chunks as they arrive
        response = ""
        for reply in test_llm:
            if reply.choices[0].delta.content:
                response += reply.choices[0].delta.content
        return response
