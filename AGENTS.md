# AGENTS.md

## Project Overview

Novel RAG is a Python-based Retrieval-Augmented Generation (RAG) web service for querying Chinese novels. It uses FastAPI, LlamaIndex, and Zhipu AI (智谱 AI) GLM models to provide streaming responses with hybrid retrieval (vector + BM25 search).

### Tech Stack

- **Python**: 3.13+
- **Web Framework**: FastAPI with Uvicorn
- **RAG Framework**: LlamaIndex (llama-index-core)
- **LLM Provider**: Zhipu AI (GLM-4.7)
- **Embedding**: Zhipu AI (embedding-3)
- **SDK**: zai-sdk
- **Logging**: Loguru
- **Configuration**: Pydantic Settings
- **Package Manager**: uv

---

## Essential Commands

### Development

```bash
# Install dependencies
uv sync

# Run locally (from src/ directory)
cd src
uv run uvicorn main:app --host 0.0.0.0 --port 8000

# Run locally (with reload)
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker Deployment

```bash
# Build and start service
docker compose up -d

# View logs
docker compose logs -f novel-rag

# Stop service
docker compose down

# Rebuild after changes
docker compose up -d --build
```

### API Testing

```bash
# Health check
curl http://localhost:8000/

# Chat endpoint (streaming)
curl -X POST http://localhost:8000/chat \
  -H "Authorization: your_access_password" \
  -H "Content-Type: application/json" \
  -d '{"chat_id": "", "question": "最后驾驶自然选择号的舰长是谁？"}'
```

---

## Project Structure

```
novel_rag/
├── src/                    # Source code directory
│   ├── main.py            # FastAPI application with /chat endpoint
│   ├── config.py          # Configuration management (pydantic-settings)
│   ├── rag.py             # RAGService, ZhipuLLM, ZhipuEmbedding classes
│   └── lifespan.py        # FastAPI lifespan manager
├── original_document/     # Raw text files for indexing (Chinese novels)
├── storage_index/         # Persistent vector storage (generated, not in git)
├── index.html             # Simple web UI for chatting
├── pyproject.toml         # Project dependencies and config
├── uv.lock               # Dependency lock file
├── Dockerfile            # Container image definition
├── docker-compose.yml    # Docker orchestration
├── .env.example          # Environment variable template
└── .env                  # Environment variables (not committed)
```

---

## Code Organization

### Main Components

#### `main.py` - FastAPI Application
- Health check endpoint (`GET /`)
- Chat endpoint with streaming (`POST /chat`)
- Simple password-based authentication via `Authorization` header
- CORS enabled for all origins
- Uses SSE (Server-Sent Events) for streaming responses

#### `config.py` - Settings Management
- Uses `pydantic-settings.BaseSettings` for environment variable configuration
- Settings are loaded from `../.env` (relative to src/)
- Configuration validation on module import
- Key settings: `API_KEY`, `PASSWORD`, `DEBUG`, `IS_GLOBAL`

#### `rag.py` - RAG Core Logic
- **ZhipuLLM**: Custom `CustomLLM` implementation using zai-sdk
  - Handles chat, stream_chat, complete, stream_complete
  - Uses `object.__setattr__` to bypass Pydantic field validation for client
  - Implements `_get_client()` to handle client recreation after serialization
  - Filters unsupported LLM parameters
- **ZhipuEmbedding**: Custom `BaseEmbedding` implementation
  - Implements all embedding methods (sync/async)
  - Uses `object.__setattr__` pattern for client management
  - Model: `embedding-3`
- **RAGService**: Main service class with singleton pattern
  - Initializes LLM and embedding models
  - Hybrid retrieval: Vector + BM25 with Reciprocal Rank Fusion
  - Streaming query support
  - Persistent index storage

#### `lifespan.py` - Application Lifecycle
- Initializes RAG service on startup
- Thread-safe initialization with double-checked locking
- Logs service readiness status

---

## Key Patterns and Conventions

### 1. Custom LLM/Embedding Implementation

The project implements custom LLM and Embedding classes for LlamaIndex:

```python
class ZhipuLLM(CustomLLM):
    # Use object.__setattr__ for private attributes to bypass Pydantic
    object.__setattr__(self, '_client', ZhipuAiClient(api_key=api_key))

    # Implement _get_client() to handle client recreation after serialization
    def _get_client(self):
        if self._client is None:
            from zai import ZhipuAiClient
            self._client = ZhipuAiClient(api_key=self._api_key)
        return self._client
```

**Why**: LlamaIndex may serialize objects, causing the client to be lost. The `_get_client()` pattern ensures the client is recreated when needed.

### 2. Singleton Pattern with Thread Safety

RAGService uses a singleton with double-checked locking:

```python
_rag_service: RAGService | None = None
_rag_service_lock = threading.Lock()

def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        with _rag_service_lock:
            if _rag_service is None:
                _rag_service = RAGService()
    return _rag_service
```

**Why**: Ensures only one instance of RAGService exists and initialization is thread-safe.

### 3. Hybrid Retrieval Strategy

The service uses a hybrid retrieval combining:

- **Vector retriever** (dense, semantic search): `similarity_top_k=10`
- **BM25 retriever** (sparse, keyword search): `similarity_top_k=10`
- **Fusion**: Reciprocal Rank Fusion (RRF) algorithm
- **Weights**: Dense:Sparse = 0.5:0.5
- **Final results**: `similarity_top_k=15`

```python
hybrid_retriever = QueryFusionRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    mode=FUSION_MODES.RECIPROCAL_RANK,
    similarity_top_k=15,
    retriever_weights=[0.5, 0.5],
)
```

### 4. Streaming Responses

FastAPI uses SSE for streaming:

```python
async def generate_response():
    async for token in rag_service.query_stream(request.question):
        yield json.dumps({"data": f"{token}"}) + '\n\n'

return StreamingResponse(generate_response(), media_type="text/event-stream")
```

### 5. System Prompt Enforcement

The RAG service enforces strict context-based responses to prevent hallucination:

- LLM must answer only from provided context
- Must explicitly state if context is insufficient
- Must not use external training knowledge
- Follows structured response format: 【回答】【相关信息】【信息说明】

### 6. Configuration Management

Settings use `pydantic-settings` with case-sensitive environment variables:

```python
class Settings(BaseSettings):
    API_KEY: str = ""
    PASSWORD: str = ""

    model_config = SettingsConfigDict(
        env_file="../.env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )
```

**Note**: The `env_file` path is relative to `src/` directory.

---

## Important Gotchas

### 1. Relative Paths

The RAG service uses relative paths from `src/`:

```python
self.original_path = '../original_document'
self.persist_path = '../storage_index'
```

**When running locally**: Must be in `src/` directory.

**When running in Docker**: Working directory is `/app/src`, so paths resolve correctly.

### 2. Index Persistence

- Index is stored in `storage_index/` directory
- This directory is **not committed** to git (in `.gitignore`)
- On first run, the index is built from documents (may take minutes)
- On subsequent runs, the index is loaded from storage (fast)

**To rebuild index**: Delete `storage_index/` directory and restart.

### 3. Environment Variables

- Required: `API_KEY` (Zhipu AI), `PASSWORD` (access control)
- Optional: `DEBUG` (default: `True`), `IS_GLOBAL` (default: `False`)
- Must create `.env` file from `.env.example` template
- Docker reads `.env` from project root, not from `src/`

### 4. Authentication

- Simple header-based authentication
- Uses `Authorization` header with password as value
- Returns 401 if password doesn't match `config_settings.PASSWORD`

### 5. Document Chunking

- Chunk size: 1024 characters
- Chunk overlap: 256 characters
- Separator: `\n`
- Documents are automatically chunked using LlamaIndex's `SentenceSplitter`

### 6. Streaming Token Handling

The streaming response extracts text from different token types:

```python
# Token can be str, CompletionResponse, or have .text or .content attributes
if isinstance(token, str):
    content = token
elif hasattr(token, 'text'):
    content = token.text
elif hasattr(token, 'content'):
    content = token.content
```

### 7. Client Serialization Issue

LlamaIndex may serialize the RAGService, causing the ZhipuAiClient to be `None`. Always use `_get_client()` method instead of accessing `_client` directly.

### 8. CORS Configuration

CORS is enabled for all origins (`allow_origins=["*"]`). Adjust in production if needed.

### 9. No Test Suite

The project currently has no tests. When adding features, consider adding unit tests.

### 10. Python Version

Requires Python 3.13+. Check `.python-version` file.

---

## Environment Variables

Create `.env` file in project root (copy from `.env.example`):

```bash
DEBUG=True
IS_GLOBAL=False
PASSWORD="<YOUR_PASSWORD_HERE>"

API_KEY="<YOUR_ZHIPUAI_API_KEY>"
```

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_KEY` | Yes | - | Zhipu AI API key |
| `PASSWORD` | Yes | - | Access password for API authentication |
| `DEBUG` | No | `True` | Debug mode flag |
| `IS_GLOBAL` | No | `False` | Use global node (China vs Global) |

---

## API Endpoints

### GET `/`

Health check endpoint.

**Response**: `{"message": "Welcome to Novel RAG!"}`

### POST `/chat`

Streaming chat endpoint with RAG.

**Headers**:
- `Authorization`: Password (required)

**Request Body**:
```json
{
  "chat_id": "optional-chat-id",
  "question": "Your question here"
}
```

**Response**: Server-Sent Events (SSE) streaming JSON lines:

```json
{"data": "token1"}

{"data": "token2"}

...
```

**Error Response** (401): `{"error": "Unauthorized"}`

---

## RAG Configuration

### Retrieval Parameters (in `rag.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `chunk_size` | 1024 | Document chunk size |
| `chunk_overlap` | 256 | Chunk overlap |
| `vector_top_k` | 10 | Vector retriever results |
| `bm25_top_k` | 10 | BM25 retriever results |
| `similarity_top_k` | 15 | Final fused results |
| `retriever_weights` | `[0.5, 0.5]` | Dense:Sparse weights |
| `llm_name` | `glm-4.7` | LLM model |
| `embedding_model_name` | `embedding-3` | Embedding model |

### System Prompt

The system prompt enforces strict context-based answers to prevent hallucination. Key requirements:

- Answer ONLY from provided context
- State explicitly if context is insufficient
- Do NOT use external knowledge
- Use structured format: 【回答】【相关信息】【信息说明】

---

## Development Workflow

### Using the Web UI

A simple web interface is provided at `index.html`:

1. Open `index.html` in a browser
2. Enter the access password from `.env` (PASSWORD variable)
3. Type your question
4. Click "Send" or press `Ctrl + Enter`
5. AI response will stream in with typewriter effect

**Note**: The web UI communicates with the backend at `http://localhost:8000`. Ensure the backend is running before using the UI.

### Adding New Documents

1. Place text files in `original_document/`
2. Delete `storage_index/` to force index rebuild
3. Restart service: `docker compose up -d --build`

### Modifying RAG Configuration

Edit `rag.py` parameters in `RAGService.__init__()`, then rebuild.

### Changing LLM Model

1. Update `self.llm_name` in `rag.py`
2. Rebuild: `docker compose up -d --build`

### Debugging

- Set `DEBUG=True` in `.env`
- View logs: `docker compose logs -f novel-rag`
- Loguru logs are visible in container output

---

## Docker Notes

### Image Build

- Base: `python:3.13-slim`
- Package manager: `uv` (copied from `ghcr.io/astral-sh/uv:latest`)
- Install: `uv sync --frozen --no-dev`
- Port: 8000
- Health check: `curl -f http://localhost:8000/`

### Volume Mounts

- `./storage_index:/app/storage_index` - Persistent index storage
- `./original_document:/app/original_document` - Document files

### Working Directory

`/app/src` (set in docker-compose.yml)

---

## Common Issues

### "API_KEY is not set"

- Check `.env` file exists and contains `API_KEY`
- Ensure `.env` is in project root, not `src/`

### "Index not initialized"

- Check `storage_index/` directory exists
- Check RAG service initialization logs
- Try deleting `storage_index/` and restarting

### 401 Unauthorized

- Check `Authorization` header matches `PASSWORD` in `.env`

### Slow first startup

- Index is being built from documents (normal on first run)
- Check logs for "Creating index" progress

### Import errors for zai-sdk

- Ensure `uv sync` completed successfully
- Check dependencies in `pyproject.toml`

---

## Dependencies (from pyproject.toml)

- `fastapi>=0.128.0` - Web framework
- `llama-index>=0.14.12` - RAG framework
- `llama-index-llms-openai>=0.6.12` - LLM integration
- `llama-index-embeddings-openai>=0.5.0` - Embedding integration
- `llama-index-retrievers-bm25>=0.6.5` - BM25 retrieval
- `loguru>=0.7.3` - Logging
- `openai>=1.60.0` - OpenAI-compatible API client
- `pydantic-settings>=2.12.0` - Configuration
- `python-dotenv>=1.2.1` - Environment variables
- `uvicorn>=0.40.0` - ASGI server
- `zai-sdk>=0.2.0` - Zhipu AI SDK

---

## Future Enhancements to Consider

- Add unit tests
- Add rate limiting
- Implement chat history storage
- Add more authentication options (JWT, OAuth)
- Add metrics/monitoring (Prometheus)
- Implement query caching
- Add multiple model support
- Implement batch document processing
