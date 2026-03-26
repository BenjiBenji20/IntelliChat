from abc import ABC, abstractmethod
from datetime import datetime
from typing import List
from uuid import UUID, uuid4
from langchain_core.documents import Document


class BaseChunker(ABC):
    
    @abstractmethod
    def chunk(
        self,
        content: str | list[dict],
        document_id: UUID,
        file_name: str,
    ) -> List[Document] | None:
        """
        Chunks content into LangChain Documents with metadata.
        
        Returns List[Document] on success.
        Returns None on unrecoverable failure — service marks document as failed.
        """
        pass

    
    def _build_document(
        self,
        file_type: str,
        content: str,
        document_id: UUID,
        chunk_id: uuid4,
        chunk_index: int,
        file_name: str,
        ingestion_time: datetime,
        document_type: str = "knowledge_base", # ex: faq, q&a
        content_type: str = "knowledge",
        section: str = None, # section title
        heading_level: int = None, # ## = 2 ### = 3
        pdf_title: str = None, # only for pdf
        page_number: int = None, # for pdf
        json_path: str = None, # ex: users[1]
        record_id: str = None # for jsonl
    ) -> Document:
        """
        Builds a single Document based on file type.
        Pass meaningful metadata according to filetype
        """
        # unified metadata dictionary. Fields that are not applicable to the file_type 
        # will naturally default to None/null instead of being missing from the payload.
        metadata = {
            "document_id": str(document_id),
            "chunk_id": str(chunk_id),
            "chunk_index": chunk_index,
            "file_name": file_name,
            "file_type": file_type,
            "content_type": content_type,
            "document_type": document_type,
            "ingestion_time": str(ingestion_time),
            
            # Type-specific fields
            "section": section,
            "heading_level": heading_level,
            "title": pdf_title,
            "page_number": page_number,
            "json_path": json_path,
            "record_id": record_id
        }
        
        return Document(
            page_content=content,
            metadata=metadata
        )
        