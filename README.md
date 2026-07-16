# Enterprise Knowledge Assistant with Hybrid Retrieval

A production-grade Retrieval-Augmented Generation (RAG) system with a strict hybrid retrieval pipeline, hallucination prevention, and a modern Next.js dashboard. 

The system extracts text from PDFs, DOCXs, and TXTs, generates dense embeddings (`all-MiniLM-L6-v2`) and sparse indices (`BM25`), reranks candidates with a cross-encoder (`ms-marco-MiniLM-L-6-v2`), and generates grounded answers via Gemini 2.0 Flash.

[![CI Status](https://github.com/your-username/your-repo/actions/workflows/test.yml/badge.svg)](https://github.com/your-username/your-repo/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Features

- **Asynchronous Document Processing**: Documents are chunked and embedded in the background, updating a SQLite UI polling state without blocking API consumers.
- **Hybrid Search**: Combines ChromaDB dense embeddings with an in-memory BM25 index (0.6 / 0.4 weight ratio, min-max normalized).
- **Semantic Reranking**: Uses a cross-encoder to strictly evaluate the top-K hybrid candidates.
- **Strict Hallucination Prevention**: Rejects queries with low-relevance scores (`< 0.3`) and validates LLM citations programmatically.
- **Fully Asynchronous Backend**: The FastAPI server wraps all blocking ML / LLM network calls in a `ThreadPool` to remain concurrent under load.

## Architecture

![Architecture Diagram](docs/architecture-diagram.md)

Detailed documentation is available in the [`docs/`](docs/) directory:
- [Architecture Details](docs/architecture.md)
- [Hybrid Retrieval Pipeline](docs/retrieval-pipeline.md)

## Tech Stack

| Component | Technology | Version |
|---|---|---|
| Frontend Framework | Next.js App Router | `15.5.20` |
| Frontend Data Fetching | React Query / Axios | `@tanstack/react-query ^5` |
| Styling | TailwindCSS | `^4` |
| Backend Framework | FastAPI | `0.111.0` (with Uvicorn) |
| Database | SQLite (`aiosqlite`) & SQLAlchemy | `2.0.30` |
| Vector Store | ChromaDB | `0.5.0` |
| RAG Models | `sentence-transformers`, `rank-bm25` | `2.7.0` |
| LLM SDK | `google-genai` | `0.1.0` |

## Folder Structure

```
├── backend/
│   ├── app/
│   │   ├── api/          # FastAPI Routes
│   │   ├── core/         # DB & Config
│   │   ├── models/       # Pydantic & SQLAlchemy Models
│   │   ├── rag/          # ML Pipeline (Vector Store, BM25, Gemini)
│   │   └── services/     # Cross-layer Orchestration
│   └── tests/
├── frontend/
│   ├── src/
│   │   ├── app/          # Next.js Pages
│   │   ├── components/   # React Components
│   │   ├── hooks/        # React Query Custom Hooks
│   │   └── __tests__/
├── docs/                 # Architecture Documentation
├── evaluation/           # RAG Evaluation Scripts & Data
├── docker-compose.yml
└── README.md
```

## Setup & Installation

### 1. Prerequisites
- [Docker & Docker Compose](https://docs.docker.com/get-docker/)

### 2. Environment Variables
Copy `.env.example` to `.env` in the root directory.
```bash
cp .env.example .env
```
Ensure you add a valid `GEMINI_API_KEY`. If no key is provided, the application will fallback to a `MockProvider` and return an error message, but the indexing/retrieval will still work.

### 3. Run the Application
Start the stack using Docker Compose:
```bash
docker compose up --build -d
```
- **Frontend Dashboard:** [http://localhost:3000](http://localhost:3000)
- **Backend API Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

## Contributing & Local Development

For faster iteration, you can run the services without Docker.

### Backend
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
Run tests:
```bash
python -m pytest tests/
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```
Run tests & linters:
```bash
npm run test
npm run lint
```

## Screenshots
*(Add screenshots here after deploying the app)*
- `docs/screenshots/dashboard.png` (Placeholder)
- `docs/screenshots/chat-view.png` (Placeholder)

## Roadmap
- [ ] Support for ingesting URLs/Confluence pages.
- [ ] Add PostgreSQL backend for scalable metadata and distributed deployments.
- [ ] Add streaming responses for the Chat API.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
