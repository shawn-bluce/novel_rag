import asyncio
from uuid import uuid4
import json
from typing import Optional
from pydantic import BaseModel

from loguru import logger
from fastapi import FastAPI, Query, Header, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from rag import get_rag_service
from lifespan import lifespan
from config import settings as config_settings

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

            # 首先使用 query_engine 获取文档引用（同步调用）
            def get_documents():
                response = rag_service.query_engine.query(request.question)
                # 提取文档引用
                sources = []
                for i, node in enumerate(response.source_nodes):
                    preview = node.node.get_content()[:200].replace("\n", " ")
                    sources.append({
                        "index": i + 1,
                        "score": float(node.score),
                        "content_preview": preview,
                        "file_name": node.node.metadata.get('file_name', 'Unknown')
                    })
                return sources, response.source_nodes

            loop = asyncio.get_event_loop()
            sources, source_nodes = await loop.run_in_executor(None, get_documents)

            # 发送文档引用信息
            yield json.dumps({"type": "sources", "data": sources}) + '\n\n'

            # 构建带上下文的提示词
            context_str = "\n\n".join([n.node.get_content() for n in source_nodes])

            # 在后台线程运行同步流式查询
            import queue
            token_queue = queue.Queue()

            def stream_producer():
                try:
                    # 使用带上下文的流式查询
                    prompt = rag_service.qa_prompt_template.partial_format(
                        system_prompt=rag_service.system_prompt,
                        context_str=context_str,
                        query_str=request.question
                    )

                    streaming_response = rag_service.llm.stream_complete(str(prompt))
                    for token in streaming_response:
                        # 检查是否有 reasoning_content（思考过程）
                        if hasattr(token, 'text'):
                            text = token.text
                        else:
                            text = str(token)
                        token_queue.put(("content", text))
                    token_queue.put(None)  # Sentinel for end of stream
                except Exception as e:
                    token_queue.put(("error", f"Error: {str(e)}"))

            # 启动后台线程
            import threading
            thread = threading.Thread(target=stream_producer)
            thread.start()

            # 从队列异步读取 token
            while True:
                try:
                    token_type, token = await loop.run_in_executor(None, token_queue.get)
                    if token is None:  # Sentinel
                        break
                    if token_type == "content":
                        logger.info(f"chat_id: {chat_data['chat_id']}, response token: {token}")
                        yield json.dumps({"type": "content", "data": token}) + '\n\n'
                    else:
                        yield json.dumps({"type": "error", "data": token}) + '\n\n'
                except Exception as e:
                    logger.error(f"Queue read error: {e}")
                    break

            thread.join()

        except Exception as e:
            logger.error(f"Query error: {e}")
            yield json.dumps({"type": "error", "data": str(e)}) + '\n\n'

    return StreamingResponse(
        generate_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
        }
    )
