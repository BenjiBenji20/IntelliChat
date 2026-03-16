import logging
from io import BytesIO
from uuid import UUID, uuid4
from datetime import datetime
from typing import List

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from doc_worker.modules.chunkings.base_chunker import BaseChunker

logger = logging.getLogger(__name__)


class PdfChunker(BaseChunker):
    """
    PDF chunker — splits by page first, then recursively
    splits oversized pages within chunk size limit.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, document_type: str = "knowledge_base"):
        self.document_type = document_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk(
        self,
        content: bytes,
        document_id: UUID,
        file_name: str,
    ) -> List[Document] | None:
        try:
            if not content or not isinstance(content, bytes):
                logger.error(
                    f"PdfChunker received invalid content type for "
                    f"document_id={document_id}, file_name={file_name}"
                )
                return None

            reader = PdfReader(BytesIO(content))

            if not reader.pages:
                logger.error(
                    f"PdfChunker received empty PDF for "
                    f"document_id={document_id}, file_name={file_name}"
                )
                return None

            # extract pdf title from metadata if available
            pdf_title = None
            if reader.metadata and reader.metadata.title:
                pdf_title = reader.metadata.title

            ingestion_time = datetime.now()
            chunks: List[Document] = []
            chunk_index = 0
            
            for page_number, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text()

                if not page_text or not page_text.strip():
                    # skip empty pages silently
                    continue

                page_text = page_text.strip()

                # if page fits within chunk size, keep as single chunk
                if len(page_text) <= self.chunk_size:
                    chunks.append(
                        self._build_document(
                            file_type="pdf",
                            content=page_text,
                            document_id=document_id,
                            document_type=self.document_type,
                            chunk_index=chunk_index,
                            chunk_id=uuid4(),
                            file_name=file_name,
                            ingestion_time=ingestion_time,
                            pdf_title=pdf_title,
                            page_number=page_number,
                        )
                    )
                    chunk_index += 1
                    
                else:
                    # page too large — split recursively
                    splits = self.splitter.split_text(page_text)
                    for split in splits:
                        chunks.append(
                            self._build_document(
                                file_type="pdf",
                                content=split,
                                document_id=document_id,
                                chunk_id=uuid4(),
                                chunk_index=chunk_index,
                                file_name=file_name,
                                ingestion_time=ingestion_time,
                                pdf_title=pdf_title,
                                page_number=page_number,
                            )
                        )
                        chunk_index+=1

            if not chunks:
                logger.error(
                    f"PdfChunker produced no chunks for "
                    f"document_id={document_id}, file_name={file_name}"
                )
                return None

            return chunks

        except Exception as e:
            logger.error(
                f"PdfChunker failed for "
                f"document_id={document_id}, file_name={file_name}. "
                f"Error: {e}"
            )
            return None
