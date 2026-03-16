"""
Chunking orchestrator
Chunk files according to file type
"""
from doc_worker.modules.chunkings.base_chunker import BaseChunker
from doc_worker.modules.chunkings.text_chunker import TextChunker
from doc_worker.modules.chunkings.pdf_chunker import PdfChunker
from doc_worker.modules.chunkings.markdown_chunker import MarkdownChunker
from doc_worker.modules.chunkings.json_chunker import JsonChunker
from doc_worker.modules.chunkings.jsonl_chunker import JsonlChunker

class ChunkerFactory:
    @staticmethod
    def get_chunker(
        file_type: str,
        document_type: str = "knowledge_base",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> BaseChunker:
        if file_type == "txt":
            return TextChunker(chunk_size, chunk_overlap, document_type)
        elif file_type == "pdf":
            return PdfChunker(chunk_size, chunk_overlap, document_type)
        elif file_type == "md":
            return MarkdownChunker(chunk_size, chunk_overlap, document_type)
        elif file_type == "json":
            return JsonChunker(document_type)
        elif file_type == "jsonl":
            return JsonlChunker(document_type)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

chunker_factory = ChunkerFactory()
