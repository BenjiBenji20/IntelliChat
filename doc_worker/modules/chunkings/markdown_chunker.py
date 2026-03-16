import logging
from uuid import UUID, uuid4
from datetime import datetime
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from doc_worker.modules.chunkings.base_chunker import BaseChunker

logger = logging.getLogger(__name__)

HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]


class MarkdownChunker(BaseChunker):
    """
    Markdown chunker — splits by headers first (H1, H2, H3).
    Oversized sections are recursively split by RecursiveCharacterTextSplitter.
    Header context is prepended to chunk content AND stored in metadata.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, document_type: str = "knowledge_base"):
        self.document_type = document_type
        self.chunk_size = chunk_size
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=HEADERS_TO_SPLIT_ON,
            strip_headers=True,  # removes header line from content, prepend it manually
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk(
        self,
        content: str,
        document_id: UUID,
        file_name: str,
    ) -> List[Document] | None:
        try:
            if not content or not isinstance(content, str):
                logger.error(
                    f"MarkdownChunker received invalid content type for "
                    f"document_id={document_id}, file_name={file_name}"
                )
                return None

            # split by headers first
            header_splits = self.header_splitter.split_text(content)

            if not header_splits:
                logger.error(
                    f"MarkdownChunker produced no splits for "
                    f"document_id={document_id}, file_name={file_name}"
                )
                return None

            ingestion_time = datetime.now()
            chunks: List[Document] = []
            chunk_index = 0

            for split in header_splits:
                # build header path from metadata
                # ex: "Introduction > Getting Started > Installation"
                header_path = self._build_header_path(split.metadata)

                # prepend header path to content for better retrieval context
                content_with_header = (
                    f"{header_path}\n{split.page_content}".strip()
                    if header_path
                    else split.page_content.strip()
                )

                if not content_with_header:
                    continue

                # section fits within chunk size — keep as single chunk
                if len(content_with_header) <= self.chunk_size:
                    chunks.append(
                        self._build_document(
                            file_type="md",
                            content=content_with_header,
                            document_type=self.document_type,
                            document_id=document_id,
                            chunk_index=chunk_index,
                            chunk_id=uuid4(),
                            file_name=file_name,
                            ingestion_time=ingestion_time,
                            section=header_path or None,
                            heading_level=self._get_deepest_heading_level(split.metadata),
                        )
                    )
                    chunk_index+=1

                # section too large — split recursively
                else:
                    sub_splits = self.text_splitter.split_text(content_with_header)
                    for sub_split in sub_splits:
                        if not sub_split.strip():
                            continue
                        chunks.append(
                            self._build_document(
                                file_type="md",
                                content=sub_split.strip(),
                                document_id=document_id,
                                document_type=self.document_type,
                                chunk_index=chunk_index,
                                chunk_id=uuid4(),
                                file_name=file_name,
                                ingestion_time=ingestion_time,
                                section=header_path or None,
                                heading_level=self._get_deepest_heading_level(split.metadata),
                            )
                        )
                        chunk_index+=1

            if not chunks:
                logger.error(
                    f"MarkdownChunker produced no chunks for "
                    f"document_id={document_id}, file_name={file_name}"
                )
                return None

            return chunks

        except Exception as e:
            logger.error(
                f"MarkdownChunker failed for "
                f"document_id={document_id}, file_name={file_name}. "
                f"Error: {e}"
            )
            return None

    # -------------------------------------------------------------------------
    # HELPER METHODS
    # -------------------------------------------------------------------------

    def _build_header_path(self, metadata: dict) -> str | None:
        """
        Builds a readable header path from split metadata.
        Example: {"Header 1": "Intro", "Header 2": "Setup"} 
            → "Intro > Setup"
        """
        if not metadata:
            return None
        path_parts = [
            metadata[key] for key in ["Header 1", "Header 2", "Header 3"]
            if key in metadata and metadata[key]
        ]
        return " > ".join(path_parts) if path_parts else None


    def _get_deepest_heading_level(self, metadata: dict) -> int | None:
        """
        Returns the deepest heading level present in metadata.
        Example: {"Header 1": "Intro", "Header 2": "Setup"} → 2
        """
        level = None
        if "Header 1" in metadata:
            level = 1
        if "Header 2" in metadata:
            level = 2
        if "Header 3" in metadata:
            level = 3
        return level
    