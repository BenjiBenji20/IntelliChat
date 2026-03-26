from doc_worker.modules.embeddings.base_embedder import BaseEmbedder
from doc_worker.modules.embeddings.cohere_embedder import CohereEmbedder
from doc_worker.modules.embeddings.gemini_embedder import GeminiEmbedder
from doc_worker.modules.embeddings.openai_embedder import OpenAIEmbedder
from shared.ai_models_details import SUPPORTED_EMBEDDING_PROVIDERS

class EmbedderFactory:

    @staticmethod
    def get_embedder(
        provider: str,
        model_name: str,
        api_key: str
    ) -> BaseEmbedder:
        prov_clean = provider.lower().strip()
        if prov_clean == "google ai studio" and prov_clean in SUPPORTED_EMBEDDING_PROVIDERS:
            return GeminiEmbedder(
                api_key=api_key,
                model_name=model_name,
                provider=provider
            )
        elif prov_clean == "openai" and prov_clean in SUPPORTED_EMBEDDING_PROVIDERS:
            return OpenAIEmbedder(
                api_key=api_key, 
                model_name=model_name,
                provider=provider
            )
        elif prov_clean == "cohere" and prov_clean in SUPPORTED_EMBEDDING_PROVIDERS:
            return CohereEmbedder(
                api_key=api_key, 
                model_name=model_name,
                provider=provider
            )
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
        