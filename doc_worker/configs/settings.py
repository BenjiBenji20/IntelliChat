from typing import Literal
from pydantic import PostgresDsn, SecretStr
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str
    DEBUG: bool = True
    ENV: Literal["dev", "prod", "test"] = "dev"
    
    CORS_DEV_ORIGIN: str = None
    CORS_PROD_ORIGIN: str = None
    
    # AI api
    LLM_API_KEY: str = None
    LLM_NAME: str = None
    
    class Config:
        env_file = "doc_worker.env"
        extra = "ignore"
    
settings = Settings()
