from uuid import UUID
import logging

from fastapi import HTTPException, status

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

from api.configs.qdrant import get_qdrant_client
from shared.vector_details import create_collection_name

logger = logging.getLogger(__name__)

class QdrantRepository:

    @property
    def client(self) -> AsyncQdrantClient:
        return get_qdrant_client()
    
    # ----------------------------------------------------------------------------------------------
    # ON DELETE DOCUMENTS, document's vectors will be deleted in Qdrant by chatbot_id, document_id
    # ----------------------------------------------------------------------------------------------
    async def delete_document_vectors(
        self, chatbot_id: UUID, document_id: UUID
    ) -> None:
        collection_name = create_collection_name(chatbot_id)
        try:
            # skip if collection doesn't exist
            exists = await self.client.collection_exists(collection_name)
            if not exists:
                logger.warning(
                    f"Collection {collection_name} does not exist, "
                    f"skipping vector deletion for document_id={document_id}"
                )
                return

            result = await self.client.delete(
                collection_name=collection_name,
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="document_id",
                                match=MatchValue(value=str(document_id))
                            )
                        ]
                    )
                )
            )

            if result.status.name != "COMPLETED":
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=(
                        f"Qdrant delete did not complete for document_id={document_id}. "
                        f"Status: {result.status}"
                    )
                )

            logger.info(
                f"Document vectors deleted in Qdrant "
                f"document_id={document_id} collection={collection_name}"
            )

        except HTTPException:
            logger.error(
                f"Failed to delete vectors for "
                f"document_id={document_id}"
            )
            raise
    
        except Exception as e:
            logger.error(
                f"Failed to delete vectors for "
                f"document_id={document_id}. Error: {e}"
            )
            raise

    # ----------------------------------------------------------------------------------------------
    # DELETE collection by collection name
    # ----------------------------------------------------------------------------------------------
    async def delete_collection(self, chatbot_id: UUID) -> bool:
        """
        Delete a Qdrant collection by chatbot_id.
        Used for cleanup when a project or chatbot is deleted.
        """
        collection_name = create_collection_name(chatbot_id)
        await self.client.delete_collection(collection_name=collection_name)
        return True


qdrant_repo = QdrantRepository()
