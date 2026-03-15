from asyncpg import Connection
from fastapi import HTTPException, status
from uuid import UUID


class DocumentWorkerRepository:
    def __init__(self, db: Connection):
        self.db = db

    async def get_document_status(self, document_id: UUID) -> str | None:
        """Get document status by id"""
        try:
            query = (
                "SELECT status "
                "FROM documents "
                "WHERE id = $1"
            )
            
            result = await self.db.fetchrow(
                query, document_id
            )
            
            if not result:
                return None
            
            return result["status"]

        except HTTPException:
            raise
        except Exception as e:
            raise e


    async def update_document_status(
        self, document_id: UUID, new_status: str = "processing"
    ) -> bool:
        """
        Update document processing status.
        Valid statuses: 'pending', 'uploaded', 'processing', 'indexed', 'failed', 'inactive'
        """
        try:
            result = await self.db.execute(
                "UPDATE documents SET status = $1 WHERE id = $2",
                new_status, document_id 
            )

            # asyncpg returns "UPDATE {n}" — check if any row was affected
            affected = int(result.split(" ")[1])
            if affected == 0:
                return False

            return True

        except HTTPException:
            raise
        except Exception as e:
            raise e
        
        
    async def get_embedding_model_details(
        self, chatbot_id: UUID 
    ) -> dict[str, str] | None:
        """Returns encrypted embedding model api key and name"""
        try:
            # extact only the rows needed
            query = (
                "SELECT api_key_encrypted, embedding_model_name, provider "
                "FROM embedding_model_keys e "
                "JOIN chatbots c ON c.embedding_model_key_id = e.id "
                "WHERE c.id = $1"
            )
            
            row = await self.db.fetchrow(
                query, chatbot_id
            )
            
            if not row:
                return None

            return {
                "api_key_encrypted": row["api_key_encrypted"],
                "embedding_model_name": row["embedding_model_name"],
                "provider": row["provider"]
            }

        except HTTPException:
            raise
        except Exception as e:
            raise e
        