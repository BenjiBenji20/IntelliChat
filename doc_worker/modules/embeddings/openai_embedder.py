import logging
from langchain_openai import OpenAIEmbeddings

from doc_worker.modules.embeddings.base_embedder import BaseEmbedder
from shared.ai_models_details import OPEN_AI_MODEL_VECTOR_MAP

logger = logging.getLogger(__name__)

class OpenAIEmbedder(BaseEmbedder):
    
    def __init__(self, api_key: str, model_name: str, provider: str):
        result = self.determine_vector_space(
            vector_name=model_name,
            model_vector_map=OPEN_AI_MODEL_VECTOR_MAP
        )
        if result is None:
            raise ValueError(
                f"Unsupported provider or model: provider={provider}, model={model_name}"
            )

        self.VECTOR_SIZE, self.VECTOR_NAME = result
        self.model_name = model_name
        self.client = OpenAIEmbeddings(
            api_key=api_key,
            model=model_name
        )
        
        
    async def embed(self, text: str) -> list[float] | None:
        try:
            vector = await self.client.aembed_query(text)
            
            if not vector:
                logger.error(
                    f"OpenAIEmbedder received empty vector "
                    f"for model={self.model_name}"
                )
                return None

            return vector
        except Exception as e:
            error_str = str(e).lower()

            # invalid key → unrecoverable
            if "api key" in error_str or "401" in error_str or "403" in error_str:
                logger.error(
                    f"OpenAIEmbedder invalid API key "
                    f"for model={self.model_name}. Error: {e}"
                )
                return None

            # recoverable → raise → Cloud Tasks retries
            logger.error(
                f"OpenAIEmbedder unexpected error "
                f"for model={self.model_name}. Error: {e}"
            )
            raise
        