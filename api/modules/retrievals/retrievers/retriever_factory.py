from qdrant_client import AsyncQdrantClient

from api.modules.retrievals.retrievers.base_retriever import BaseRetriever
from api.modules.retrievals.retrievers.gemini_retriever import GeminiRetriever
from api.modules.retrievals.retrievers.openai_retriever import OpenAIRetriever
from api.modules.retrievals.retrievers.cohere_retriever import CohereRetriever
from shared.ai_models_details import SUPPORTED_EMBEDDING_PROVIDERS

class RetrieverFactory:
    @staticmethod
    def create_retrieval(
        provider: str,
        api_key: str, # raw embedding model api_key
        model_name: str,
        qdrant: AsyncQdrantClient
    ) -> BaseRetriever:
        """Orchestrates the flow of retrieval based on provider"""
        prov_clean = provider.lower().strip()
        if prov_clean == "google ai studio" and prov_clean in SUPPORTED_EMBEDDING_PROVIDERS:
            return GeminiRetriever(api_key=api_key, model_name=model_name, qdrant=qdrant)
        elif prov_clean == "openai" and prov_clean in SUPPORTED_EMBEDDING_PROVIDERS:
            return OpenAIRetriever(provider=provider, api_key=api_key, model_name=model_name, qdrant=qdrant)
        elif prov_clean == "cohere" and prov_clean in SUPPORTED_EMBEDDING_PROVIDERS:
            return CohereRetriever(provider=provider, api_key=api_key, model_name=model_name, qdrant=qdrant)
        
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
