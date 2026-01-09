from loguru import logger
from zai import ZaiClient, ZhipuAiClient
from zai.core import APIRequestFailedError, APIAuthenticationError, APIReachLimitError, APIInternalError, APIServerFlowExceedError, APIStatusError

from config import settings


if settings.IS_GLOBAL:
    client = ZaiClient(api_key=settings.API_KEY)
else:
    client = ZhipuAiClient(api_key=settings.API_KEY)


def check_healthy():
    try:
        response = client.chat.completions.create(
            model="glm-4.5-flash",  # using free model for testing
            messages=[
                {"role": "user", "content": "I'm testing the connection, you need to response 'Connection successful' only."}
            ]
        )
        content = response.choices[0].message.content
        logger.info(f"Connection test response: {content}")
        return True
    except (APIRequestFailedError, APIAuthenticationError, APIReachLimitError,
            APIInternalError, APIServerFlowExceedError, APIStatusError) as e:
        logger.error(f"Connection check failed: {e}")
        return False