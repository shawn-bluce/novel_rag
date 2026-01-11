import os
import time
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
        self.system_prompt = 'You are a helpful assistant. Use the following context to answer the question accurately.'
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
        self.llm = ZhipuAI(model='glm-4-flash', api_key=config_settings.API_KEY)
        self.embedding_model = ZhipuAIEmbedding(model='embedding-3', api_key=config_settings.API_KEY)
        self.chunk_size = 512
        self.chunk_overlap = 32
        self.vector_top_k = 5
        self.bm25_top_k = 5
        self.similarity_top_k = 10  # 融合后返回的最终结果数量
        self.retriever_weights = [0.5, 0.5]  # Dense 和 Sparse 各占 50% 权重
        self.separator = '\n'

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


if __name__ == '__main__':
    rag_service = RAGService()
    rag_service.embedding()
    rag_service.re_order_retriever()
    test_question = "请简要介绍一下黑暗森林法则是什么？"
    answer = rag_service.query(test_question)
    print("回答:", answer)