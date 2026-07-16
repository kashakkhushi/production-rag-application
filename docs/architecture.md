# Enterprise Knowledge Assistant Architecture

The backend of the Enterprise Knowledge Assistant follows a strict layered architecture designed for testability, concurrency, and modularity. 

## Dependency Flow

The system flows strictly in one direction:
`api` → `services` → `rag`

### 1. `app.api` (Presentation Layer)
- **Role:** Handles FastAPI routing, request validation, HTTP status codes, and background task dispatching.
- **Rules:** The API layer is completely ignorant of how ML models, vector stores, or LLMs work. It delegates entirely to the `services` layer.

### 2. `app.services` (Orchestration Layer)
- **Role:** Coordinates operations between the database and the RAG pipeline.
- **Key Modules:**
  - `chat_service.py`: Safely orchestrates the `/chat` endpoint by offloading blocking CPU-bound retrieval (`ChromaDB`, `BM25`) and network I/O (`Gemini`) to threadpools using `run_in_threadpool`, ensuring the async ASGI event loop is never blocked.
  - `orchestrator.py`: Manages background indexing tasks. It coordinates extracting text via `document_processor.py`, generating embeddings, persisting to ChromaDB, atomically rebuilding the BM25 index, and updating the SQLite status.

### 3. `app.rag` (Domain Layer)
- **Role:** Encapsulates all AI, retrieval, and generation logic.
- **Key Modules:**
  - `vector_store.py`: Abstracts ChromaDB interactions. Swappable.
  - `sparse_index.py`: Manages the in-memory BM25 index. Uses atomic assignment for thread-safe asynchronous rebuilds without locking read queries.
  - `embeddings.py` / `reranker.py`: Wraps `sentence-transformers` cross-encoders and dense embeddings. Models are loaded exactly once at module initialization to prevent memory leaks and latency.
  - `generation.py`: Contains the `LLMProvider` interface. This dependency inversion allows seamless fallback to a `MockProvider` if the Gemini API key is missing, maintaining system stability and testability.

## Data Persistence
- **ChromaDB:** Persistent volume used for storing dense vector embeddings and chunk metadata.
- **SQLite (`aiosqlite`):** Stores asynchronous document lifecycle state (`processing`, `indexed`, `failed`) ensuring the UI can poll for real-time indexing status without querying the vector database.
- **BM25:** In-memory sparse index rebuilt dynamically. This ensures lightning-fast keyword retrieval while keeping the architecture simple, avoiding the need for an external Elasticsearch cluster.
