from typing import Literal
from pydantic import SecretStr
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
    
    ENCRYPTION_KEY: str = None
    
    # GCP
    CLOUD_TASK_QUEUE_PATH: str = None
    QUEUE_NAME: str = None
    WORKER_URL: str = None
    GCP_PROJECT_ID: str = None
    GCP_REGION: str = None
    GCS_BUCKET_NAME: str = None
    MAIN_API_SERVICE_ACC_EMAIL: str = None

    # Supabase
    POSTGRES_USER: str = None
    POSTGRES_PASSWORD: SecretStr = None
    POSTGRES_HOST: str = None
    POSTGRES_PORT: str = None
    POSTGRES_DATABASE: str = None
    
    QDRANT_API_KEY: str = None
    QDRANT_CLUSTER_ENDPOINT: str = None
    
    class Config:
        extra = "ignore"
        
    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql://{self.POSTGRES_USER}:"
            f"{self.POSTGRES_PASSWORD.get_secret_value()}"
            f"@{self.POSTGRES_HOST}:"
            f"{self.POSTGRES_PORT}/"
            f"{self.POSTGRES_DATABASE}?ssl=require"
        )
    
settings = Settings()
