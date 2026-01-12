import os
import time
import asyncio
import threading
from loguru import logger

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.llms import CustomLLM, CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import Field
from typing import List, Optional, Sequence
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import PromptTemplate

from config import settings as config_settings


class ZhipuLLM(CustomLLM):
    """使用 zai-sdk 的 LLM 类"""

    context_window: int = Field(default=128000, description="Context window size")
    num_output: int = Field(default=4096, description="Number of output tokens")
    model_name: str = Field(default="glm-4.7", description="Model name")

    # 显式声明为私有属性
    _model: str = ""
    _api_key: str = ""

    def __init__(self, model: str = "glm-4.7", api_key: str = "", **kwargs):
        super().__init__(model_name=model, **kwargs)
        # 使用 object.__setattr__ 绕过 Pydantic 的字段管理
        object.__setattr__(self, '_model', model)
        object.__setattr__(self, '_api_key', api_key)
        object.__setattr__(self, '_client', None)
        # 导入 zai SDK
        try:
            from zai import ZhipuAiClient
            object.__setattr__(self, '_client', ZhipuAiClient(api_key=api_key))
        except ImportError:
            raise ImportError("请安装 zai-sdk: pip install zai-sdk")

    def _get_client(self):
        """确保客户端存在，处理序列化后丢失的情况"""
        if self._client is None:
            try:
                from zai import ZhipuAiClient
                self._client = ZhipuAiClient(api_key=self._api_key)
            except ImportError:
                raise ImportError("请安装 zai-sdk: pip install zai-sdk")
        return self._client

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    def chat(self, messages: Sequence[dict], **kwargs) -> CompletionResponse:
        client = self._get_client()
        # 转换消息格式
        formatted_messages = []
        for msg in messages:
            if hasattr(msg, 'role'):
                formatted_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
            else:
                formatted_messages.append(msg)

        # 过滤掉不支持的参数
        supported_params = {
            'temperature', 'top_p', 'max_tokens', 'stream', 'stop',
            'presence_penalty', 'frequency_penalty', 'user', 'n'
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported_params}

        response = client.chat.completions.create(
            model=self._model,
            messages=formatted_messages,
            **filtered_kwargs
        )
        return CompletionResponse(text=response.choices[0].message.content)

    def stream_chat(self, messages: Sequence[dict], **kwargs) -> CompletionResponseGen:
        client = self._get_client()
        # 转换消息格式
        formatted_messages = []
        for msg in messages:
            if hasattr(msg, 'role'):
                formatted_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
            else:
                formatted_messages.append(msg)

        # 过滤掉不支持的参数
        supported_params = {
            'temperature', 'top_p', 'max_tokens', 'stream', 'stop',
            'presence_penalty', 'frequency_penalty', 'user', 'n'
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported_params}

        response = client.chat.completions.create(
            model=self._model,
            messages=formatted_messages,
            stream=True,
            **filtered_kwargs
        )

        def gen():
            for chunk in response:
                logger.debug(f"Stream chunk: {chunk}")
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    # 检查 content 字段
                    content = delta.content if hasattr(delta, 'content') else None
                    # 检查 reasoning_content 字段（思维链）
                    reasoning = delta.reasoning_content if hasattr(delta, 'reasoning_content') else None

                    logger.debug(f"Delta: content={content}, reasoning={reasoning}")

                    # 输出 reasoning_content（思考过程），用特殊标记包装
                    if reasoning is not None:
                        yield CompletionResponse(text=f"<thinking>{reasoning}</thinking>")
                    # 输出 content（实际回答）
                    elif content is not None:
                        yield CompletionResponse(text=content)

        return gen()

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)

    def stream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
        messages = [{"role": "user", "content": prompt}]
        return self.stream_chat(messages, **kwargs)


class ZhipuEmbedding(BaseEmbedding):
    """使用 zai-sdk 的 Embedding 类"""

    # 显式声明为私有属性，避免 Pydantic 验证
    _model: str = ""
    _api_key: str = ""

    def __init__(
        self,
        model: str = "embedding-3",
        api_key: str = "",
        api_base: str = "https://open.bigmodel.cn/api/paas/v4/",
        **kwargs
    ):
        super().__init__(**kwargs)
        # 使用 object.__setattr__ 绕过 Pydantic 的字段管理
        object.__setattr__(self, '_model', model)
        object.__setattr__(self, '_api_key', api_key)
        object.__setattr__(self, '_api_base', api_base)
        object.__setattr__(self, '_client', None)
        # 导入 zai SDK 并创建客户端
        try:
            from zai import ZhipuAiClient
            object.__setattr__(self, '_client', ZhipuAiClient(api_key=api_key))
        except ImportError:
            raise ImportError("请安装 zai-sdk: pip install zai-sdk")

    def _get_client(self):
        """确保客户端存在，处理序列化后丢失的情况"""
        if self._client is None:
            try:
                from zai import ZhipuAiClient
                self._client = ZhipuAiClient(api_key=self._api_key)
            except ImportError:
                raise ImportError("请安装 zai-sdk: pip install zai-sdk")
        return self._client

    @classmethod
    def class_name(cls) -> str:
        return "ZhipuEmbedding"

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._get_text_embeddings(texts)

    def _get_query_embedding(self, query: str) -> List[float]:
        client = self._get_client()
        response = client.embeddings.create(
            model=self._model,
            input=query,
        )
        return response.data[0].embedding

    def _get_text_embedding(self, text: str) -> List[float]:
        client = self._get_client()
        response = client.embeddings.create(
            model=self._model,
            input=text,
        )
        return response.data[0].embedding

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        client = self._get_client()
        response = client.embeddings.create(
            model=self._model,
            input=texts,
        )
        return [data.embedding for data in response.data]

    def get_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._get_text_embeddings(texts)


class RAGService:
    def __init__(self):
        self.system_prompt = '''
        你是一个专业、准确的智能问答助手。你的任务是**仅基于提供的上下文信息**回答用户的问题。

        ## 核心原则
        
        ### 1. 严格基于上下文
        - **只使用提供的上下文片段**来回答问题
        - 如果上下文中没有相关信息，**明确告知用户**"根据提供的资料，没有找到相关信息"
        - **绝对不要编造、猜测或使用外部知识**来补充答案
        
        ### 2. 准确引用信息
        - 当答案来自多个上下文片段时，**综合这些信息**给出完整的回答
        - 如果上下文片段之间有冲突，**如实说明**不同版本的描述
        - **引用具体的细节和事实**，而不是模糊的概括
        
        ### 3. 清晰的答案结构
        - 使用清晰、简洁的语言
        - 对于复杂问题，可以分点作答
        - 如果需要推理，**展示推理过程**，但推理必须基于上下文
        
        ### 4. 诚实面对信息不足
        - 如果上下文信息不足以完整回答问题，**诚实说明**
        - 可以总结上下文中的相关信息，然后**指出缺失的部分**
        - 不要为了"给出答案"而强行编造
        
        ### 5. 保持客观中立
        - 客观陈述上下文中的信息
        - 避免添加个人观点或情感色彩
        - 如果上下文中有不同立场或观点，**公平呈现**
        
        ## 回答格式
        
        ```
        【回答】
        [基于上下文的准确回答]
        
        【相关信息】
        [总结引用的上下文片段的关键信息]
        
        【信息说明】(可选)
        [如果信息不完整或需要补充说明，在此处说明]
        ```
        
        ## 示例
        
        **示例 1：信息充足**
        用户：什么是"掩体计划"？
        上下文：提供了关于掩体计划的详细描述
        回答：
        【回答】
        掩体计划是《三体》中人类高层为了应对三体人入侵而制定的战略计划之一...
        
        【相关信息】
        根据提供的资料，掩体计划的核心内容包括：[具体描述]
        
        **示例 2：信息不完整**
        用户：掩体计划和黑域计划的详细技术参数对比是什么？
        上下文：只提供了两个计划的概述，没有详细技术参数
        回答：
        【回答】
        根据提供的资料，关于掩体计划和黑域计划的技术参数对比，文档中只提到了基本概念，没有提供详细的技术参数对比信息。
        
        【相关信息】
        掩体计划：[上下文中的描述]
        黑域计划：[上下文中的描述]
        
        【信息说明】
        提供的资料中没有包含这两个计划的详细技术参数、具体实施步骤或量化对比数据。
        
        ## 严格禁止
        - ❌ 编造上下文中没有的事实
        - ❌ 使用训练数据中的外部知识来补充答案
        - ❌ 模糊其辞，假装回答了实际问题
        - ❌ 混淆上下文信息和外部知识
        
        记住：**准确和诚实比完整更重要**。宁可承认信息不足，也不要编造答案。
        '''
        self.qa_prompt_template = PromptTemplate(
            template="""{system_prompt}
        
        上下文信息：
        ---------------------
        {context_str}
        ---------------------
        
        查询：{query_str}
        
        回答：""",
        )


        self.original_path = '../original_document'
        self.persist_path = '../storage_index'
        self.llm_name = 'glm-4.7'
        self.embedding_model_name = 'embedding-3'

        # 使用自定义的 ZhipuLLM
        self.llm = ZhipuLLM(
            model=self.llm_name,
            api_key=config_settings.API_KEY,
        )
        # 使用自定义的 ZhipuEmbedding
        self.embedding_model = ZhipuEmbedding(
            model=self.embedding_model_name,
            api_key=config_settings.API_KEY,
        )
        self.chunk_size = 1024       # document chunk size
        self.chunk_overlap = 256     # document chunk overlap
        self.vector_top_k = 10       # count of vector retriever top k
        self.bm25_top_k = 10         # count of bm25 retriever top k
        self.similarity_top_k = 15   # count of merged retriever top k
        self.retriever_weights = [0.5, 0.5]  # Dense : Sparse
        self.separator = '\n'       # document chunk separator

        self.index = None
        self.query_engine = None

    def embedding(self):
        if not os.path.exists(self.persist_path):
            Settings.embed_model = self.embedding_model
            # 设置文本分割器
            Settings.text_splitter = SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separator=self.separator,
            )
            logger.info("Local index not found. Creating new index...")
            documents = SimpleDirectoryReader(self.original_path).load_data()
            logger.info(f"Loaded {len(documents)} documents from {self.original_path}")

            logger.info("Creating index, maybe take some minutes...")
            start_time = time.time()
            self.index = VectorStoreIndex.from_documents(documents)
            end_time = time.time()
            logger.success(f"Index created in {end_time - start_time:.2f} seconds.")
            logger.info("Persisting index to local storage...")
            start_time = time.time()
            self.index.storage_context.persist(persist_dir=self.persist_path)
            end_time = time.time()
            logger.success(f"Index persisted in {end_time - start_time:.2f} seconds.")
        else:
            logger.info("Loading index from local storage...")
            start_time = time.time()
            Settings.embed_model = self.embedding_model
            storage_context = StorageContext.from_defaults(persist_dir=self.persist_path)
            self.index = load_index_from_storage(storage_context)
            end_time = time.time()
            logger.success(f"Index loaded in {end_time - start_time:.2f} seconds.")

    def re_order_retriever(self):
        if self.index is None:
            logger.error("Index not initialized. Please run embedding() first.")
            return

        vector_retriever = self.index.as_retriever(similarity_top_k=self.vector_top_k)
        bm25_retriever = BM25Retriever.from_defaults(index=self.index, similarity_top_k=self.bm25_top_k)
        hybrid_retriever = QueryFusionRetriever(
            retrievers=[
                vector_retriever,  # [0] Dense 检索器 - 语义理解
                bm25_retriever,  # [1] Sparse 检索器 - 关键词匹配
            ],
            # 使用 Reciprocal Rank Fusion 算法融合结果
            mode=FUSION_MODES.RECIPROCAL_RANK,  # RRF 融合算法（重排序）
            # 最终返回的节点数量
            similarity_top_k=self.similarity_top_k,
            num_queries=1,  # 不进行查询扩展（1=原查询）
            retriever_weights=self.retriever_weights,  # Dense 和 Sparse 各占 50% 权重
            llm=self.llm,  # 显式指定使用智谱 GLM 模型
        )

        self.query_engine = RetrieverQueryEngine.from_args(
            retriever=hybrid_retriever,  # 使用混合检索器
            node_postprocessors=[
                # 可选：添加相似度后处理器过滤低相关性结果
                # SimilarityPostprocessor(similarity_cutoff=0.1)  # 过滤相似度 < 0.3 的文档
            ],
            # 使用 LLM 进行最终答案生成
            llm=self.llm,
            # 应用自定义的 System Prompt
            text_qa_template=self.qa_prompt_template.partial_format(system_prompt=self.system_prompt),
            # 启用流式输出
            streaming=True,
        )

    def query(self, question: str) -> str:
        if self.query_engine is None:
            logger.error("Query engine not initialized. Please run re_order_retriever() first.")
            return "Query engine not initialized."

        response = self.query_engine.query(question)
        logger.info("[引用来源]:")
        for i, node in enumerate(response.source_nodes):
            preview = node.node.get_content()[:100].replace("\n", " ")
            logger.debug(f"  来源 {i + 1} (Score={node.score:.4f}): ...{preview}...")

        return response

    async def query_stream(self, question: str):
        """流式查询，返回异步生成器"""
        if self.query_engine is None:
            logger.error("Query engine not initialized. Please run re_order_retriever() first.")
            yield "Query engine not initialized."
            return

        # 使用异步流式查询
        streaming_response = await self.query_engine.aquery(question)
        logger.debug(f"Streaming response type: {type(streaming_response)}")
        logger.debug(f"Has async_response_gen: {hasattr(streaming_response, 'async_response_gen')}")

        # AsyncStreamingResponse 对象使用 async_response_gen() 方法
        if hasattr(streaming_response, 'async_response_gen'):
            async for token in streaming_response.async_response_gen():
                logger.debug(f"Token type: {type(token)}, token: {token}")
                logger.debug(f"Token dir: {dir(token)}")

                # 提取 CompletionResponse 的 text 属性
                if hasattr(token, 'text'):
                    content = token.text
                    logger.debug(f"Token.text = '{content}'")
                elif isinstance(token, str):
                    content = token
                    logger.debug(f"Token is str: '{content}'")
                else:
                    content = str(token)
                    logger.debug(f"Token str(): '{content}'")

                # 只 yield 非空内容
                if content:
                    logger.debug(f"Yielding content: '{content}'")
                    yield content
                else:
                    logger.debug("Skipping empty token")
        else:
            # 如果不是流式对象，直接返回结果（兼容旧版本）
            yield str(streaming_response)


_rag_service: RAGService | None = None
_rag_service_lock = threading.Lock()
_initialized = False


def get_rag_service() -> RAGService:
    """获取 RAG 服务单例实例（线程安全）"""
    global _rag_service, _initialized

    if _rag_service is None:
        with _rag_service_lock:
            # 双重检查锁定模式
            if _rag_service is None:
                logger.info("Creating RAG service singleton instance...")
                _rag_service = RAGService()

    return _rag_service


def initialize_rag_service():
    """初始化 RAG 服务（应在应用启动时调用一次）"""
    global _initialized

    rag_service = get_rag_service()

    if not _initialized:
        with _rag_service_lock:
            if not _initialized:
                logger.info("Initializing RAG service...")
                rag_service.embedding()
                rag_service.re_order_retriever()
                _initialized = True
                logger.success("RAG service initialized successfully.")

    return rag_service


def is_rag_service_ready() -> bool:
    """检查 RAG 服务是否已初始化"""
    return _initialized