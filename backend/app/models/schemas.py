from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class DocumentResponse(BaseModel):
    id: str
    filename: str
    status: str
    upload_timestamp: datetime
    chunk_count: int
    error_message: Optional[str] = None

class UploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    message: str

class ChatRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    
class Citation(BaseModel):
    document_name: str
    page_number: int
    chunk_id: str
    score: float

class RetrievalStats(BaseModel):
    retrieval_latency_ms: float
    retrieved_chunks_count: int
    dense_scores: List[float]
    sparse_scores: List[float]
    reranker_scores: List[float]

class ChatResponse(BaseModel):
    answer: str
    sources: List[Citation]
    confidence: str # high|medium|low
    retrieval_stats: RetrievalStats
