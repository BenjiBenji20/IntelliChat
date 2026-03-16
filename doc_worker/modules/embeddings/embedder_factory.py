from doc_worker.modules.embeddings.base_embedder import BaseEmbedder
from doc_worker.modules.embeddings.gemini_embedder import GeminiEmbedder


SUPPORTED_PROVIDERS = {
    "Google AI Studio"
    # Add more providers here
}

class EmbedderFactory:

    @staticmethod
    def get_embedder(
        provider: str,
        model_name: str,
        api_key: str
    ) -> BaseEmbedder:
        if provider == "Google AI Studio" and provider in SUPPORTED_PROVIDERS:
            return GeminiEmbedder(
                api_key=api_key,
                model_name=model_name,
                provider=provider
            )
        # future:
        # elif provider == "OpenAI" and provider in SUPPORTED_PROVIDERS:
        #     return OpenAIEmbedder(api_key=api_key, model_name=model_name)
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
        