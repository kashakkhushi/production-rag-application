import os
import shutil
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List

from app.core.database import get_db
from app.models.domain import DocumentDB
from app.models.schemas import UploadResponse, DocumentResponse, ChatRequest, ChatResponse, RetrievalStats
from app.services.orchestrator import process_document_background, delete_document
from app.services.chat_service import process_chat_query
from app.core.config import settings
from app.rag.vector_store import vector_store
from app.rag.sparse_index import sparse_index
from app.rag.generation import llm_provider, MockProvider

router = APIRouter()

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    if not file.filename.endswith(('.pdf', '.docx', '.txt')):
        raise HTTPException(status_code=400, detail="Only PDF, DOCX, and TXT files are supported.")
        
    doc = DocumentDB(filename=file.filename)
    db.add(doc)
    await db.commit()
    await db.refresh(doc)
    
    # Save file temporarily
    temp_path = f"/tmp/{doc.id}_{file.filename}"
    # Ensure /tmp exists (usually does in docker, but safe check)
    os.makedirs("/tmp", exist_ok=True)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Start background task
    background_tasks.add_task(process_document_background, doc.id, temp_path, file.filename)
    
    return UploadResponse(
        document_id=doc.id,
        filename=file.filename,
        status="processing",
        message="Upload successful, processing in background."
    )

@router.get("/documents", response_model=List[DocumentResponse])
async def get_documents(skip: int = 0, limit: int = 100, db: AsyncSession = Depends(get_db)):
    stmt = select(DocumentDB).offset(skip).limit(limit)
    result = await db.execute(stmt)
    docs = result.scalars().all()
    return docs

@router.delete("/document/{doc_id}")
async def delete_document_endpoint(doc_id: str, db: AsyncSession = Depends(get_db)):
    doc = await db.get(DocumentDB, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
        
    await delete_document(doc_id)
    return {"message": "Document deleted successfully"}

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    top_k = request.top_k or settings.TOP_K
    return await process_chat_query(request.query, top_k)

@router.get("/health")
async def health_check():
    health_status = {
        "status": "ok",
        "chromadb": "ok",
        "sqlite": "ok",
        "gemini": "ok",
        "embedding_model": "ok",
        "cross_encoder": "ok",
        "bm25_initialized": "ok"
    }
    
    # Check Chroma
    try:
        vector_store.client.heartbeat()
    except Exception as e:
        health_status["chromadb"] = f"error: {str(e)}"
        health_status["status"] = "error"
        
    # Check Gemini
    if isinstance(llm_provider, MockProvider):
        health_status["gemini"] = "warning: LLM provider not configured"
        
    # Check BM25
    if sparse_index.bm25 is None and len(sparse_index.corpus_chunks) > 0:
        # Should be initialized if we have chunks
        health_status["bm25_initialized"] = "error: missing initialization"
        
    return health_status
