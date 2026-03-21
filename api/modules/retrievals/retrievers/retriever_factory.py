from qdrant_client import AsyncQdrantClient

from api.modules.retrievals.retrievers.base_retriever import BaseRetriever
from api.modules.retrievals.retrievers.gemini_retriever import GeminiRetriever
from shared.ai_models_details import SUPPORTED_PROVIDERS

class RetrieverFactory:
    @staticmethod
    def create_retrieval(
        provider: str,
        api_key: str, # raw embedding model api_key
        model_name: str,
        qdrant: AsyncQdrantClient
    ) -> BaseRetriever:
        """Orchestrates the flow of retrieval based on provider"""
        if provider.lower().strip() == "google ai studio" and provider in SUPPORTED_PROVIDERS:
            return GeminiRetriever(api_key=api_key, model_name=model_name, qdrant=qdrant)
        # future:
        # elif provider.lower().strip() == "openai" and provider in SUPPORTED_PROVIDERS:
        #     return OpenAIRetriever(provider=provider, api_key=api_key, model_name=model_name, qdrant=qdrant)
        
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
