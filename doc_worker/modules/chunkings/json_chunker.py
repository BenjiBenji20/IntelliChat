import json
import logging
from uuid import UUID
from uuid import uuid4
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
                    f"JsonChunker received invalid content type for "
                    f"document_id={document_id}, file_name={file_name}"
                )
                return None

            ingestion_time = datetime.now()
            chunks: List[Document] = []

            for chunk_index, record in enumerate(content):
                
                if not isinstance(record, (dict, list)):
                    logger.warning(
                        f"JsonChunker skipping non-dict record at index={chunk_index} "
                        f"document_id={document_id}"
                    )
                    continue
                
                # serialize entire nested object as one chunk
                serialized = json.dumps(record, ensure_ascii=False, indent=2)
                
                # array chunk
                if isinstance(record, list):
                    # construct json_path for metadata
                    json_path = f"item[{chunk_index}]"

                # object chunk → json_path = first string value in object
                elif isinstance(record, dict):
                    # get first key's value as json_path identifier
                    first_key = next(iter(record))
                    first_value = record[first_key]
                    json_path = str(first_value) if isinstance(first_value, (str, int)) else f"[{chunk_index}]"
                    
                chunks.append(
                    self._build_document(
                        file_type="json",
                        content=serialized,
                        document_id=document_id,
                        document_type=self.document_type,
                        chunk_index=chunk_index,
                        chunk_id=uuid4(),
                        file_name=file_name,
                        ingestion_time=ingestion_time,
                        json_path=json_path,
                    )
                )
                        
            if not chunks:
                logger.error(
                    f"JsonChunker produced no chunks for "
                    f"document_id={document_id}, file_name={file_name}"
                )
                return None

            return chunks

        except Exception as e:
            logger.error(
                f"JsonChunker failed for "
                f"document_id={document_id}, file_name={file_name}. "
                f"Error: {e}"
            )
            return None
        