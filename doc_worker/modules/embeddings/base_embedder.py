from abc import ABC, abstractmethod


class BaseEmbedder(ABC):

    VECTOR_NAME: str   # e.g. "gemini-embedding-001"
    VECTOR_SIZE: int   # e.g. 3072
    
    @abstractmethod
    async def embed(self, text: str) -> list[float] | None:
        """
        Embeds a single text string.

        Returns list[float] on success.
        Returns None on unrecoverable failure (invalid key, bad response).
        Raises on recoverable failure (API down, timeout) → Cloud Tasks retries.
        """
        pass
    