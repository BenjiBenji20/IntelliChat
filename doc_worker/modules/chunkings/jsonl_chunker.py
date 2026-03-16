import json
import logging
from uuid import UUID, uuid4
from datetime import datetime
from typing import List

from langchain_core.documents import Document
from doc_worker.modules.chunkings.base_chunker import BaseChunker

logger = logging.getLogger(__name__)

# common id field names to look for
ID_FIELD_CANDIDATES = {"id", "ID", "Id", "_id", "uuid", "record_id", "key"}


class JsonlChunker(BaseChunker):
    """
    JSONL chunker — each line/record becomes one chunk.
    Traces record_id from object keys if available.
    """
    def __init__(self, document_type: str = "knowledge_base"):
        self.document_type = document_type
        pass

    def chunk(
        self,
        content: list[dict],
        document_id: UUID,
        file_name: str,
    ) -> List[Document] | None:
        try:
            if not content or not isinstance(content, list):
                logger.error(
                    f"JsonlChunker received invalid content type for "
                    f"document_id={document_id}, file_name={file_name}"
                )
                return None

            ingestion_time = datetime.now()
            chunks: List[Document] = []

            for record in content:
                index = uuid4()
                
                if not isinstance(record, dict):
                    logger.warning(
                        f"JsonlChunker skipping non-dict record at index={index} "
                        f"document_id={document_id}"
                    )
                    continue

                # trace record_id from object keys
                record_id = self._extract_record_id(record)

                serialized = json.dumps(record, ensure_ascii=False, indent=2)

                chunks.append(
                    self._build_document(
                        file_type="jsonl",
                        content=serialized,
                        document_id=document_id,
                        document_type=self.document_type,
                        chunk_index=index,
                        file_name=file_name,
                        ingestion_time=ingestion_time,
                        record_id=record_id,
                    )
                )

            if not chunks:
                logger.error(
                    f"JsonlChunker produced no chunks for "
                    f"document_id={document_id}, file_name={file_name}"
                )
                return None

            return chunks

        except Exception as e:
            logger.error(
                f"JsonlChunker failed for "
                f"document_id={document_id}, file_name={file_name}. "
                f"Error: {e}"
            )
            return None

    def _extract_record_id(self, record: dict) -> str | None:
        """
        Traces record_id by scanning object keys.
        Example: {"id": "qa-general-pricing-hm", ...} → "qa-general-pricing-hm"
        Returns None if no id field found.
        """
        for key in record.keys():
            if key in ID_FIELD_CANDIDATES:
                value = record[key]
                # only return if value is a meaningful string or int
                if isinstance(value, (str, int)) and value:
                    return str(value)
        return None
    