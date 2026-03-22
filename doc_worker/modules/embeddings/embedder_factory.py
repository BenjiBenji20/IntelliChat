from doc_worker.modules.embeddings.base_embedder import BaseEmbedder
from doc_worker.modules.embeddings.gemini_embedder import GeminiEmbedder
from shared.ai_models_details import SUPPORTED_EMBEDDING_PROVIDERS

class EmbedderFactory:

    @staticmethod
    def get_embedder(
        provider: str,
        model_name: str,
        api_key: str
    ) -> BaseEmbedder:
        if provider == "google ai studio" and provider in SUPPORTED_EMBEDDING_PROVIDERS:
            return GeminiEmbedder(
                api_key=api_key,
                model_name=model_name,
                provider=provider
            )
        # future:
        # elif provider == "OpenAI" and provider in SUPPORTED_EMBEDDING_PROVIDERS:
        #     return OpenAIEmbedder(api_key=api_key, model_name=model_name)
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
        