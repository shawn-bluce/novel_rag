import os
import time
import asyncio
import threading
from loguru import logger

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.llms.zhipuai import ZhipuAI
from llama_index.embeddings.zhipuai import ZhipuAIEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import PromptTemplate

from config import settings as config_settings


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
        self.llm_name = 'glm-4-flash'
        self.embedding_model_name = 'embedding-3'
        self.llm = ZhipuAI(model=self.llm_name, api_key=config_settings.API_KEY)
        self.embedding_model = ZhipuAIEmbedding(model=self.embedding_model_name, api_key=config_settings.API_KEY)
        self.chunk_size = 512       # document chunk size
        self.chunk_overlap = 32     # document chunk overlap
        self.vector_top_k = 5       # count of vector retriever top k
        self.bm25_top_k = 5         # count of bm25 retriever top k
        self.similarity_top_k = 10  # count of merged retriever top k
        self.retriever_weights = [0.5, 0.5]  # Dense : Sparse
        self.separator = '\n'       # document chunk separator

        self.index = None
        self.query_engine = None

    def embedding(self):
        if not os.path.exists(self.persist_path):
            Settings.embed_model = self.embedding_model
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

        # 使用流式查询
        streaming_response = await self.query_engine.aquery(question)

        # AsyncStreamingResponse 对象使用 async_response_gen() 方法
        if hasattr(streaming_response, 'async_response_gen'):
            async for token in streaming_response.async_response_gen():
                yield token
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