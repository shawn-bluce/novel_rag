import asyncio
from uuid import uuid4
import json
from typing import Optional
from pydantic import BaseModel

from loguru import logger
from fastapi import FastAPI, Query, Header, Request
from fastapi.responses import StreamingResponse, JSONResponse
from rag import get_rag_service
from lifespan import lifespan
from config import settings as config_settings

app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    chat_id: str = ""
    question: str = ""


@app.get("/")
async def say_hello():
    return {"message": f"Welcome to {config_settings.APP_NAME}!"}

@app.post("/chat")
async def chat(
        request: ChatRequest,
        authorization: Optional[str] = Header(None)
):
    logger.debug(f"Authorization (via Header): {authorization}")
    if authorization != config_settings.PASSWORD:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})

    chat_data = {
        "chat_id": request.chat_id if request.chat_id else str(uuid4()),
        "message": {
            "role": "user",
            "content": request.question
        }
    }

    async def generate_response():
        try:
            rag_service = get_rag_service()
            async for token in rag_service.query_stream(request.question):
                # SSE or plain text
                logger.info(f"chat_id: {chat_data['chat_id']}, response token: {token}")
                yield json.dumps({"data": f"{token}"}) + '\n\n'
        except Exception as e:
            logger.error(f"Query error: {e}")
            yield f"\n[Error: {str(e)}]"

    return StreamingResponse(
        generate_response(),
        media_type="text/event-stream"
    )
