from api.modules.chat.llm.chat_anthropic import ChatAnthropic
from api.modules.chat.llm.chat_google import ChatGoogle
from api.modules.chat.llm.chat_groq import ChatGroq
from api.modules.chat.llm.chat_openai import ChatOpenAI
from api.modules.chat.llm.chat_xai import ChatXAI
from shared.ai_models_details import llm_provider_mapper, get_llm_context_window
from api.modules.chat.llm.base_llm import BaseLLM

class LLMFactory:
    @staticmethod
    def create_llm(
        model_name: str,
        api_key: str, #  raw llm api_key
        provider: str,
    ) -> BaseLLM:
        """
        Orchestrates the flow of LLM chat based on provider.
        This helps to use correct model SDK.
        """
        
        if not llm_provider_mapper(model_name=model_name, provider=provider):
            raise ValueError(
                f"Model '{model_name}' is not supported under provider '{provider}'."
            )
        
        prov_clean = provider.lower().strip()
        
        if prov_clean == "groq":
            llm = ChatGroq(api_key=api_key, model_name=model_name)
            llm.context_window = get_llm_context_window(model_name, provider)
            return llm
        
        if prov_clean == "openai":
            llm = ChatOpenAI(api_key=api_key, model_name=model_name)
            llm.context_window = get_llm_context_window(model_name, provider)
            return llm
        
        if prov_clean == "google":
            llm = ChatGoogle(api_key=api_key, model_name=model_name)
            llm.context_window = get_llm_context_window(model_name, provider)
            return llm
        
        if prov_clean == "anthropic":
            llm = ChatAnthropic(api_key=api_key, model_name=model_name)
            llm.context_window = get_llm_context_window(model_name, provider)
            return llm
        
        if prov_clean == "xai":
            llm = ChatXAI(api_key=api_key, model_name=model_name)
            llm.context_window = get_llm_context_window(model_name, provider)
            return llm
        
        raise ValueError(f"Unsupported LLM provider: '{provider}'")
    