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
    API_KEY_HEADER_NAME: str | None = "Anonymous"
    
    SECRET_KEY_PREFIX: str | None = None
    ENCRYPTION_KEY: str | None = None
    
    UPSTASH_REDIS_URL: str | None = None
    UPSTASH_REDIS_TOKEN: str | None = None
    
    # AI api
    LLM_API_KEY: str | None = None
    LLM_NAME: str | None = None
    
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
    
settings = Settings()
