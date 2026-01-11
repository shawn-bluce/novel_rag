import asyncio
import json

from loguru import logger
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from rag import get_rag_service
from lifespan import lifespan

app = FastAPI(lifespan=lifespan)


async def generate_text():
    words = ["hello", "world", " from", " FastAPI"]
    for word in words:
        await asyncio.sleep(0.5)
        yield f"{word}\n\n"


@app.get("/")
async def hello():
    return StreamingResponse(
        generate_text(),
        media_type="text/event-stream"
    )


@app.get("/query")
async def query(
        question: str = Query(..., description="查询问题")
):
    async def generate_response():
        try:
            rag_service = get_rag_service()
            async for token in rag_service.query_stream(question):
                # SSE or plain text
                yield json.dumps({"data": f"{token}"}) + '\n\n'
        except Exception as e:
            logger.error(f"Query error: {e}")
            yield f"\n[Error: {str(e)}]"

    return StreamingResponse(
        generate_response(),
        media_type="text/event-stream"
    )
