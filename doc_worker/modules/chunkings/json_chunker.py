import json
import logging
from uuid import UUID
from datetime import datetime
from typing import List

from langchain_core.documents import Document
from doc_worker.modules.chunkings.base_chunker import BaseChunker

logger = logging.getLogger(__name__)


class JsonChunker(BaseChunker):
    """
    JSON chunker — each top-level record becomes one chunk.
    Serializes entire nested object as one chunk.
    """

    def chunk(
        self,
        content: list[dict],
        document_id: UUID,
        source: str,
    ) -> List[Document] | None:
        try:
            if not content or not isinstance(content, list):
                logger.error(
                    f"JsonChunker received invalid content type for "
                    f"document_id={document_id}, source={source}"
                )
                return None

            ingestion_time = datetime.now()
            chunks: List[Document] = []

            for index, record in enumerate(content):
                if not isinstance(record, (dict, list)):
                    logger.warning(
                        f"JsonChunker skipping non-dict record at index={index} "
                        f"document_id={document_id}"
                    )
                    continue
                
                # serialize entire nested object as one chunk
                serialized = json.dumps(record, ensure_ascii=False, indent=2)
                
                # array chunk
                if isinstance(record, list):
                    # construct json_path for metadata
                    json_path = f"item[{index}]"

                # object chunk → json_path = first string value in object
                elif isinstance(record, dict):
                    # get first key's value as json_path identifier
                    first_key = next(iter(record))
                    first_value = record[first_key]
                    json_path = str(first_value) if isinstance(first_value, (str, int)) else f"[{index}]"
                    
                chunks.append(
                    self._build_document(
                        file_type="json",
                        content=serialized,
                        document_id=document_id,
                        chunk_index=index,
                        source=source,
                        ingestion_time=ingestion_time,
                        json_path=json_path,
                    )
                )
                        
            if not chunks:
                logger.error(
                    f"JsonChunker produced no chunks for "
                    f"document_id={document_id}, source={source}"
                )
                return None

            return chunks

        except Exception as e:
            logger.error(
                f"JsonChunker failed for "
                f"document_id={document_id}, source={source}. "
                f"Error: {e}"
            )
            return None
        