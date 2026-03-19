import logging
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from doc_worker.modules.embeddings.base_embedder import BaseEmbedder
from shared.ai_models_details import GEMINI_MODEL_VECTOR_MAP, GOOGLE_AI_PROVIDERS

logger = logging.getLogger(__name__)


class GeminiEmbedder(BaseEmbedder):

    def __init__(self, api_key: str, model_name: str, provider: str):
        result = self.determine_vector_space(
            vector_name=model_name,
            provider=provider
        )
        if result is None:
            raise ValueError(
                f"Unsupported provider or model: provider={provider}, model={model_name}"
            )

        self.VECTOR_SIZE, self.VECTOR_NAME = result
        self.model_name = model_name
        self.client = GoogleGenerativeAIEmbeddings(
            google_api_key=api_key,
            model=model_name
        )

    @staticmethod
    def determine_vector_space(
        vector_name: str,
        provider: str
    ) -> tuple[int, str] | None:
        """
        Determines vector size and name based on provider and model name.
        Returns None if provider or model is unsupported.
        """
        if provider.lower().strip() not in GOOGLE_AI_PROVIDERS:
            return None

        model = vector_name.lower().strip()

        if model not in GEMINI_MODEL_VECTOR_MAP:
            return None

        size = GEMINI_MODEL_VECTOR_MAP[model]
        return size, model


    async def embed(self, text: str) -> list[float] | None:
        try:
            vector = await self.client.aembed_query(text)

            if not vector:
                logger.error(
                    f"GeminiEmbedder received empty vector "
                    f"for model={self.model_name}"
                )
                return None

            return vector

        except Exception as e:
            error_str = str(e).lower()

            # invalid key → unrecoverable
            if "api key" in error_str or "401" in error_str or "403" in error_str:
                logger.error(
                    f"GeminiEmbedder invalid API key "
                    f"for model={self.model_name}. Error: {e}"
                )
                return None

            # recoverable → raise → Cloud Tasks retries
            logger.error(
                f"GeminiEmbedder unexpected error "
                f"for model={self.model_name}. Error: {e}"
            )
            raise
        