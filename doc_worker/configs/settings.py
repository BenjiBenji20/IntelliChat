from typing import Literal
from pydantic import SecretStr
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str
    DEBUG: bool = True
    ENV: Literal["dev", "prod", "test"] = "dev"
    
    CORS_DEV_ORIGIN: str | None = None
    CORS_PROD_ORIGIN: str | None = None
    
    # AI api
    LLM_API_KEY: str | None = None
    LLM_NAME: str | None = None
    
    ENCRYPTION_KEY: str | None = None
    
    # GCP
    CLOUD_TASK_QUEUE_PATH: str | None = None
    QUEUE_NAME: str | None = None
    WORKER_URL: str | None = None
    GCP_PROJECT_ID: str | None = None
    GCP_REGION: str | None = None
    GCS_BUCKET_NAME: str | None = None
    MAIN_API_SERVICE_ACC_EMAIL: str | None = None

    # Supabase
    POSTGRES_USER: str | None = None
    POSTGRES_PASSWORD: SecretStr | None = None
    POSTGRES_HOST: str | None = None
    POSTGRES_PORT: str | None = None
    POSTGRES_DATABASE: str | None = None
    
    QDRANT_API_KEY: str | None = None
    QDRANT_CLUSTER_ENDPOINT: str | None = None
    
    # GOOGLE CLOUD STORAGE SERVICE KEY
    TYPE: str | None = None 
    PROJECT_ID: str | None = None 
    PRIVATE_KEY_ID: str | None = None 
    PRIVATE_KEY: str | None = None 
    CLIENT_EMAIL: str | None = None 
    CLIENT_ID: str | None = None 
    AUTH_URI: str | None = None 
    TOKEN_URI: str | None = None 
    AUTH_PROVIDER_X509_CERT_URL: str | None = None 
    CLIENT_X509_CERT_URL: str | None = None 
    UNIVERSE_DOMAIN: str | None = None

    
    class Config:
        #env_file = "doc_worker.env"
        env_file = ".env" # point to native .env file name for successful cloud run deployment
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
    
    @property
    def get_gcs_credentials(self) -> dict:
        """Returns the Google credentials as a dictionary"""
        return {
            "type": self.TYPE,
            "project_id": self.PROJECT_ID,
            "private_key_id": self.PRIVATE_KEY_ID,
            "private_key": self.PRIVATE_KEY.replace("\\n", "\n"),
            "client_email": self.CLIENT_EMAIL,
            "client_id": self.CLIENT_ID,
            "auth_uri": self.AUTH_URI,
            "token_uri": self.TOKEN_URI,
            "auth_provider_x509_cert_url": self.AUTH_PROVIDER_X509_CERT_URL,
            "client_x509_cert_url": self.CLIENT_X509_CERT_URL,
            "universe_domain": self.UNIVERSE_DOMAIN
        }
    
settings = Settings()
