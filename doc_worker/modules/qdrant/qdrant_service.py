import logging
from uuid import UUID
import uuid
from typing import List

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    FilterSelector,
    Filter,
    FieldCondition,
    MatchValue,
)
from langchain_core.documents import Document

from doc_worker.configs.qdrant import get_qdrant_client
from shared.vector_details import create_collection_name

logger = logging.getLogger(__name__)

class QdrantService:

    @property
    def client(self) -> AsyncQdrantClient:
        return get_qdrant_client()

    # -------------------------------------------------------------------------
    # COLLECTION MANAGEMENT
    # -------------------------------------------------------------------------
    async def ensure_collection_exists(
        self, chatbot_id: UUID, vector_name: str, vector_size: int = 3072
    ) -> None:
        """
        Creates collection if it doesn't exist.
        Safe to call on every upsert — no error if already exists.
        Recoverable: raises on failure → Cloud Tasks retries.
        """
        collection_name = create_collection_name(chatbot_id)
        try:
            exists = await self.client.collection_exists(collection_name)
            if not exists:
                await self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        vector_name: VectorParams(
                            size=vector_size,
                            distance=Distance.COSINE,
                        )
                    }
                )
                logger.info(
                    f"QdrantService created collection: {collection_name}"
                )
            # Always ensure payload index exists for document_id
            # Required for delete_document_vectors filter to work
            await self.client.create_payload_index(
                collection_name=collection_name,
                field_name="document_id",
                field_schema="keyword"
            )
            logger.info(
                f"QdrantService ensured payload index on document_id "
                f"for collection={collection_name}"
            )
        except Exception as e:
            logger.error(
                f"QdrantService failed to ensure collection: {collection_name}. "
                f"Error: {e}"
            )
            raise  # recoverable → Cloud Tasks retries

    # -------------------------------------------------------------------------
    # IDEMPOTENCY GUARDRAIL
    # -------------------------------------------------------------------------
    async def delete_document_vectors(
        self, chatbot_id: UUID, document_id: UUID
    ) -> None:
        """
        Deletes all existing vectors for a document_id before upsert.
        Guarantees clean slate on retry — prevents duplicate vectors.
        Recoverable: raises on failure → Cloud Tasks retries.
        """
        collection_name = create_collection_name(chatbot_id)
        try:
            await self.client.delete(
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
            logger.info(
                f"QdrantService deleted existing vectors for "
                f"document_id={document_id} in collection={collection_name}"
            )
        except Exception as e:
            logger.error(
                f"QdrantService failed to delete vectors for "
                f"document_id={document_id}. Error: {e}"
            )
            raise  # recoverable → Cloud Tasks retries

    # -------------------------------------------------------------------------
    # UPSERT
    # -------------------------------------------------------------------------
    async def upsert_document_chunks(
        self,
        vector_name: str,
        chatbot_id: UUID,
        document_id: UUID,
        chunks: List[Document],
        vectors: List[List[float]],
    ) -> None:
        """
        Upserts document chunks with their vectors into Qdrant.
        Recoverable: raises on failure → Cloud Tasks retries.
        """
        collection_name = create_collection_name(chatbot_id)
        try:
            points = [
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{document_id}_{index}")), # deterministic per document+chunk
                    vector={
                        vector_name: vector
                    },
                    payload={
                        **chunk.metadata,          # all metadata from chunker
                        "page_content": chunk.page_content,
                    }
                )
                for index, (chunk, vector) in enumerate(zip(chunks, vectors))
            ]

            await self.client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True  # wait for upsert to complete before returning
            )
            logger.info(
                f"QdrantService upserted {len(points)} points for "
                f"document_id={document_id} in collection={collection_name}"
            )
        except Exception as e:
            logger.error(
                f"QdrantService failed to upsert vectors for "
                f"document_id={document_id}. Error: {e}"
            )
            raise  # recoverable → Cloud Tasks retries

qdrant = QdrantService()
