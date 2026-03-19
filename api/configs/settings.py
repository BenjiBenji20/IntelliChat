from typing import Literal
from pydantic import PostgresDsn, SecretStr
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str
    DEBUG: bool = True
    ENV: Literal["dev", "prod", "test"] = "dev"
    
    CORS_DEV_ORIGIN: str | None = None
    CORS_PROD_ORIGIN: str | None = None
    
    # db settings
    # temporarily nullable for successfully deployment in GCP cloud run 
    POSTGRES_USER: str | None = None
    POSTGRES_PASSWORD: SecretStr | None = None
    POSTGRES_HOST: str | None = None
    POSTGRES_PORT: int | None = None
    POSTGRES_DATABASE: str | None = None
    
    # api secret key security
    API_KEY_HEADER_NAME: str = "Anonymous"
    
    SECRET_KEY_PREFIX: str | None = None
    ENCRYPTION_KEY: str | None = None
    
    UPSTASH_REDIS_URL: str | None = None
    UPSTASH_REDIS_TOKEN: str | None = None
    
    # AI api
    LLM_API_KEY: str | None = None
    LLM_NAME: str | None = None
    
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
    
    GCS_BUCKET_NAME: str | None = None
    GCS_UPLOAD_URL_EXPIRY_SECONDS: int = 900
    GCS_DOWNLOAD_URL_EXPIRY_SECONDS: int = 3600
    MAX_FILE_SIZE_BYTES: int = 52428800
    
    QUEUE_NAME: str | None = None
    CLOUD_TASKS_QUEUE_PATH: str | None = None
    CLOUD_TASKS_BUCKET_SA_EMAIL: str | None = None
    WORKER_URL: str | None = None
    
    QDRANT_API_KEY: str | None = None
    QDRANT_CLUSTER_ENDPOINT: str | None = None
    
    class Config:
        env_file = ".env"
        extra = "ignore"

    @property
    def DATABASE_URL(self) -> PostgresDsn:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:"
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
            "private_key": self.PRIVATE_KEY,
            "client_email": self.CLIENT_EMAIL,
            "client_id": self.CLIENT_ID,
            "auth_uri": self.AUTH_URI,
            "token_uri": self.TOKEN_URI,
            "auth_provider_x509_cert_url": self.AUTH_PROVIDER_X509_CERT_URL,
            "client_x509_cert_url": self.CLIENT_X509_CERT_URL,
            "universe_domain": self.UNIVERSE_DOMAIN
        }
    
settings = Settings()
