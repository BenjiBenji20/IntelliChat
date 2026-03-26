SUPPORTED_EMBEDDING_PROVIDERS = {
    'google ai studio',
    'openai',
    'cohere'
}

# supported gemini embedding models and their vector sizes
GEMINI_MODEL_VECTOR_MAP = {
    "gemini-embedding-001":           3072,
    "text-embedding-004":             768,
    "gemini-embedding-2-preview":     3072,
}

OPEN_AI_MODEL_VECTOR_MAP = {
    "text-embedding-ada-002": 1536, # min=1536
    "text-embedding-3-small": 1536, # min=1
    "text-embedding-3-large": 3072 # min=1
}

COHERE_MODEL_VECTOR_MAP = {
    "embed-v4.0": 1536, # min=256
    "embed-english-v3.0": 1024, # min=fixed
    "embed-english-light-v3.0": 384, # min=fixed
    "embed-multilingual-v3.0": 1024, # min=fixed
}

# support Embedder models based on providers
# libraries available in langchain
def embedder_provider_mapper(model_name: str, provider: str) -> bool:
    EMBEDDER_MODELS_PROVIDER_MAP = {
        "google ai studio": [
            "gemini-embedding-001",
            "text-embedding-004",
            "gemini-embedding-2-preview",
        ],
        "openai": [
            "text-embedding-ada-002",
            "text-embedding-3-small",
            "text-embedding-3-large",
        ],
        "cohere": [
            "embed-v4.0",
            "embed-english-v3.0",
            "embed-english-light-v3.0",
            "embed-multilingual-v3.0",
        ]
    }
    
    prov = [
        provider for prov in EMBEDDER_MODELS_PROVIDER_MAP.keys()
        if provider.lower().strip() == prov.lower()
        and model_name.lower().strip() in EMBEDDER_MODELS_PROVIDER_MAP[prov]
    ]
    
    return True if prov else False


# Global map for provider context limits
LLM_MODELS_PROVIDER_MAP = {
    "groq": {
        "openai/gpt-oss-120b": 131_072,          # Max: 131,072 ctx
        "openai/gpt-oss-20b": 131_072,           # Max: 131,072 ctx
        "llama-3.3-70b-versatile": 131_072,      # Max: 131,072 ctx
        "llama-3.1-8b-instant": 131_072,         # Max: 131,072 ctx
        "whisper-large-v3": 8192,
        "whisper-large-v3-turbo": 8192
    },
    "openai": {
        "gpt-4o": 128_000,                       # Max: 128,000 ctx
        "gpt-4o-mini": 128_000,                  # Max: 128,000 ctx
        "gpt-4.1": 1_000_000,                    # Max: 1,000,000 ctx
        "gpt-4.1-mini": 1_000_000,               # Max: 1,000,000 ctx
        "gpt-4.1-nano": 1_000_000,               # Max: 1,000,000 ctx
        "o3": 200_000,                           # Max: 200,000 ctx
        "o4-mini": 200_000,                      # Max: 200,000 ctx
    },
    "anthropic": {
        "claude-opus-4-6": 200_000,              # Max: 200,000 ctx
        "claude-sonnet-4-6": 200_000,            # Max: 200,000 ctx
        "claude-haiku-4-5-20251001": 200_000,    # Max: 200,000 ctx
    },
    "google": {
        "gemini-2.5-pro-preview-03-25": 1_000_000,
        "gemini-2.5-flash": 1_000_000,
        "gemini-2.0-flash": 1_000_000,
        "gemini-3.1-pro-preview": 1_000_000,
    },
    "xai": {
        "grok-2-1212": 128_000,                  # Max: 128,000 ctx
        "grok-2-vision-1212": 128_000,           # Max: 128,000 ctx
        "grok-beta": 128_000,                    # Max: 128,000 ctx
    },
}

# support LLM models based on providers
# libraries available in langchain
def llm_provider_mapper(model_name: str, provider: str) -> bool:
    provider_map = LLM_MODELS_PROVIDER_MAP.get(provider.lower().strip(), {})
    return model_name.lower().strip() in provider_map


def get_llm_context_window(model_name: str, provider: str) -> int:
    """Returns the exact context window based on the model and provider maps, default fallback to 8192."""
    provider_map = LLM_MODELS_PROVIDER_MAP.get(provider.lower().strip(), {})
    return provider_map.get(model_name.lower().strip(), 8192)


def llm_provider_validator(provider: str) -> bool:
    return provider.lower().strip() in LLM_MODELS_PROVIDER_MAP.keys()
