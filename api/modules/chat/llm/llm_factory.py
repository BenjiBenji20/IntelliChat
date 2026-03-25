from api.modules.chat.llm.chat_groq import ChatGroq
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
        
        if provider.lower().strip() == "groq":
            llm = ChatGroq(api_key=api_key, model_name=model_name)
            llm.context_window = get_llm_context_window(model_name, provider)
            return llm
        
        # in future:
        # if provider == "OpenAI":
        #     return ChatOpenAI(api_key=api_key, model_name=model_name)
        
        raise ValueError(f"Unsupported LLM provider: '{provider}'")