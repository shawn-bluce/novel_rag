from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from loguru import logger


class Settings(BaseSettings):
    APP_NAME: str = "Novel RAG"
    DEBUG: bool = True

    API_KEY: str = ""
    EMBEDDING_MODEL: str = "embedding-3"
    LLM_MODEL: str = "glm-4.5-flash"
    IS_GLOBAL: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


def validate_settings() -> None:
    if not settings.API_KEY:
        logger.error("API_KEY is not set in the environment variables.")
        raise ValueError("API_KEY must be set.")
    logger.info(f"Using embedding model: {settings.EMBEDDING_MODEL}")
    logger.info(f"Using LLM model: {settings.LLM_MODEL}")
    if settings.IS_GLOBAL:
        logger.info("Running in global mode.")
    else:
        logger.info("Running in China mode.")
    logger.info(f"Debug mode is {'on' if settings.DEBUG else 'off'}.")


settings = Settings()
validate_settings()
