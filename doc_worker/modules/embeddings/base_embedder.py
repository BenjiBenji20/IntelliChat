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
    
    
    def determine_vector_space(
        self,
        vector_name: str,
        model_vector_map: dict[str, int]
    ) -> tuple[int, str] | None:
        """
        Determines vector size and name based on provider and model name.
        Returns None if provider or model is unsupported.
        """
        model = vector_name.lower().strip()

        if model not in model_vector_map:
            return None

        size = model_vector_map[model]
        return size, model
    