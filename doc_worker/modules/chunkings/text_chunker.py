import logging
from uuid import UUID
from datetime import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

from doc_worker.modules.chunkings.base_chunker import BaseChunker

logger = logging.getLogger(__name__)

class TextChunker(BaseChunker):
    """Text file chunker"""
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk(
        self,
        content: str,
        document_id: UUID,
        source: str,
    ) -> List[Document] | None:
        try:
            if not content or not isinstance(content, str):
                logger.error(
                    f"TextChunker received invalid content type for "
                    f"document_id={document_id}, source={source}"
                )
                return None
            
            splits = self.splitter.split_text(content)

            if not splits:
                logger.error(
                    f"TextChunker produced no chunks for "
                    f"document_id={document_id}, source={source}"
                )
                return None

            return [
                self._build_document(
                    content=split,
                    file_type="txt",
                    chunk_index=index,
                    document_id=document_id,
                    source=source,
                    ingestion_time=datetime.now().isoformat()
                )
                for index, split in enumerate(splits)
            ]
            
        except Exception as e:
            logger.error(
                f"TextChunker failed for "
                f"document_id={document_id}, source={source}. "
                f"Error: {e}"
            )
            return None
    