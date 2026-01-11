from contextlib import asynccontextmanager

from loguru import logger
from fastapi import FastAPI
from rag import initialize_rag_service, is_rag_service_ready


@asynccontextmanager
async def lifespan(app: FastAPI):
    """application lifespan context manager"""

    # on startup
    logger.info("Starting application...")

    # init RAG service
    try:
        initialize_rag_service()
        if is_rag_service_ready():
            logger.success("RAG service is ready for queries.")
        else:
            logger.warning("RAG service initialization may have issues.")
    except Exception as e:
        logger.error(f"Failed to initialize RAG service: {e}")
        raise

    yield  # App running......

    # on shutdown
    logger.info("Shutting down application...")
