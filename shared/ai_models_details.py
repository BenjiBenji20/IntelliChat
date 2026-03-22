SUPPORTED_EMBEDDING_PROVIDERS = {
    'google ai studio',
    'openai',
    'cohere'
}

GOOGLE_AI_PROVIDERS = {"google ai studio", "google", "gemini", "google ai"}

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


# support LLM models based on providers
# libraries available in langchain
def llm_provider_mapper(model_name: str, provider: str) -> bool:
    LLM_MODELS_PROVIDER_MAP = {
        "groq": [
            "openai/gpt-oss-120b",          # Max: 131,072 ctx / 65,536 completion
            "openai/gpt-oss-20b",           # Max: 131,072 ctx / 65,536 completion — fast & cheap
            "llama-3.3-70b-versatile",      # Max: 131,072 ctx / 32,768 completion
            "llama-3.1-8b-instant",         # Max: 131,072 ctx / 131,072 completion — very fast
            "whisper-large-v3",
            "whisper-large-v3-turbo"
        ],
        "openai": [
            "gpt-4o",                       # Max: 128,000 ctx / 16,384 output
            "gpt-4o-mini",                  # Max: 128,000 ctx / 16,384 output
            "gpt-4.1",                      # Max: 1,000,000 ctx / 32,768 output
            "gpt-4.1-mini",                 # Max: 1,000,000 ctx / 32,768 output
            "gpt-4.1-nano",                 # Max: 1,000,000 ctx / 32,768 output — cheapest
            "o3",                           # Max: 200,000 ctx — high reasoning
            "o4-mini",                      # Max: 200,000 ctx — fast reasoning
        ],
        "anthropic": [
            "claude-opus-4-6",              # Max: 200,000 ctx / 32,000 output — most capable
            "claude-sonnet-4-6",            # Max: 200,000 ctx / 16,000 output — balanced
            "claude-haiku-4-5-20251001",    # Max: 200,000 ctx / 8,192 output — fast & cheap
        ],
        "google": [
            "gemini-2.5-pro-preview-03-25", # Max: 1,000,000 ctx — most capable
            "gemini-2.5-flash",             # Max: 1,000,000 ctx — fast, cost-efficient
            "gemini-2.0-flash",             # Max: 1,000,000 ctx — free tier available
            "gemini-3.1-pro-preview",       # Max: 1,000,000 ctx — latest flagship
        ],
        "xai": [
            "grok-2-1212",                  # Max: 128,000 ctx
            "grok-2-vision-1212",           # Max: 128,000 ctx — multimodal
            "grok-beta",                    # Max: 128,000 ctx
        ],
    }
    
    prov = [
        provider for prov in LLM_MODELS_PROVIDER_MAP.keys()
        if provider.lower().strip() == prov.lower()
        and model_name.lower().strip() in LLM_MODELS_PROVIDER_MAP[prov]
    ]
    
    return True if prov else False


def llm_provider_validator(provider: str) -> bool:
    PROVIDER_SET = {
        "openai",
        "groq",
    }
    
    return True if provider.lower().strip() in PROVIDER_SET else False
