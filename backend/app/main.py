import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.router import router as api_router
from app.core.database import init_db
from app.core.exceptions import RAGException, rag_exception_handler
from app.rag.sparse_index import sparse_index

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure data dir exists for sqlite/chroma
    os.makedirs("/app/data", exist_ok=True)
    
    # Initialize DB
    await init_db()
    
    # Initialize BM25 from Chroma
    await sparse_index.rebuild_index()
    
    yield

app = FastAPI(
    title="Enterprise Knowledge Assistant with Hybrid Retrieval",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(RAGException, rag_exception_handler)

app.include_router(api_router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
