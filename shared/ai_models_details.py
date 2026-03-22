SUPPORTED_PROVIDERS = {
    'google ai studio',
    'openai',
    'anthropic',
    'azure openai'
}

GOOGLE_AI_PROVIDERS = {"google ai studio", "google", "gemini", "google ai"}

# supported gemini embedding models and their vector sizes
GEMINI_MODEL_VECTOR_MAP = {
    "gemini-embedding-001":           3072,
    "text-embedding-005":             768,
    "text-multilingual-embedding-002": 768,
    "gemini-embedding-2-preview":     3072,
}

# support Embedder models based on providers
# libraries available in langchain
def embedder_provider_mapper(model_name: str, provider: str) -> bool:
    EMBEDDER_MODELS_PROVIDER_MAP = {
        "google ai studio": [
            "gemini-embedding-001",
            "text-embedding-005",
            "text-multilingual-embedding-002",
            "gemini-embedding-exp-03-07",
        ],
        "openai": [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        ],
        "azure openai": [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        ],
    }
    
    prov = [
        provider for prov in EMBEDDER_MODELS_PROVIDER_MAP.keys()
        if provider.lower().strip() == prov.lower()
        and model_name.lower().strip() in EMBEDDER_MODELS_PROVIDER_MAP[prov]
    ]
    
    return True if prov else False

def embedder_provider_validator(provider: str) -> bool:
    PROVIDER_SET = {
        'google ai studio',
        'openai',
        'anthropic',
        'azure openai'
    }
    
    return True if provider.lower().strip() in PROVIDER_SET else False


# support LLM models based on providers
# libraries available in langchain
def llm_provider_mapper(model_name: str, provider: str) -> bool:
    LLM_MODELS_PROVIDER_MAP = {
        "OpenAI": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1",
            "gpt-4.1-mini"
        ],
        "Groq": [
            "openai/gpt-oss-120b",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768"
        ]
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
