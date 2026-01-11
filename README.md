# Novel RAG

基于智谱 AI 的小说检索增强生成（RAG）服务。支持向量检索和 BM25 检索的混合召回。

## 功能特性

- 混合检索：向量检索 + BM25 关键词检索
- 流式响应：支持实时返回生成结果
- 简单认证：基于密码的访问控制
- Docker 部署：一键启动

## 快速开始

### 1. 配置环境变量

创建 `.env` 文件：

```bash
API_KEY=your_zhipuai_api_key
PASSWORD=your_access_password
DEBUG=True
IS_GLOBAL=False
```

### 2. 准备文档

将小说文本文件放入 `original_document/` 目录。

### 3. 启动服务

```bash
docker compose up -d
```

服务将在 `http://localhost:8000` 启动。

## API 使用

### 健康检查

```bash
curl http://localhost:8000/
```

### 对话接口

```bash
curl -X POST http://localhost:8000/chat \
  -H "Authorization: your_access_password" \
  -H "Content-Type: application/json" \
  -d '{
    "chat_id": "",
    "question": "最后驾驶自然选择号的舰长是谁？",
  }'
```

## 配置说明

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| `API_KEY` | 智谱 AI API 密钥 | 必填 |
| `PASSWORD` | 访问密码 | 必填 |
| `DEBUG` | 调试模式 | `True` |
| `IS_GLOBAL` | 是否使用全球节点 | `False` |

## 目录结构

```
novel_rag/
├── original_document/   # 原始小说文档
├── storage_index/       # 持久化的向量索引
├── src/                 # 源代码
├── Dockerfile
├── docker-compose.yml
└── .env                 # 环境变量配置
```
