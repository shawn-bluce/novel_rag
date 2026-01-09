from loguru import logger
from fastapi import FastAPI

from config import settings
from zhipu_utils import check_healthy


app = FastAPI()

if check_healthy():
    logger.success("Connection to Zhipu AI API successful.")
else:
    logger.error("Failed to connect to Zhipu AI API.")

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
