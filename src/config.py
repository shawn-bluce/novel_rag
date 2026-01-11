from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from loguru import logger


class Settings(BaseSettings):
    APP_NAME: str = "Novel RAG"
    DEBUG: bool = True
    IS_GLOBAL: bool = False

    API_KEY: str = ""

    model_config = SettingsConfigDict(
        env_file="../.env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


def validate_settings() -> bool:
    logger.info(f"Debug mode is {'on' if settings.DEBUG else 'off'}.")
    if settings.IS_GLOBAL:
        logger.info("Running in global mode.")
    else:
        logger.info("Running in China mode.")

    if not settings.API_KEY:
        logger.error("API_KEY is not set. Please set it in the environment variables or .env file.")
        return False
    return True

settings = Settings()
validate_settings()
