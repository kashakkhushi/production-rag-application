from typing import Dict, Any, Tuple, List
from fastapi.concurrency import run_in_threadpool
from app.rag.pipeline import run_hybrid_retrieval
from app.rag.generation import generate_answer
from app.models.schemas import ChatResponse, RetrievalStats, Citation
from loguru import logger

async def process_chat_query(query: str, top_k: int) -> ChatResponse:
    try:
        # Run retrieval in threadpool since it calls blocking ML models (Chroma, BM25, Cross-Encoder)
        chunks, stats = await run_in_threadpool(run_hybrid_retrieval, query, top_k)
        
        # Run generation in threadpool since it calls blocking network requests
        answer, citations, confidence = await run_in_threadpool(generate_answer, query, chunks)
        
        retrieval_stats = RetrievalStats(
            retrieval_latency_ms=stats["retrieval_latency_ms"],
            retrieved_chunks_count=stats["retrieved_chunks_count"],
            dense_scores=stats["dense_scores"],
            sparse_scores=stats["sparse_scores"],
            reranker_scores=stats["reranker_scores"]
        )
        
        return ChatResponse(
            answer=answer,
            sources=citations,
            confidence=confidence,
            retrieval_stats=retrieval_stats
        )
    except Exception as e:
        logger.error(f"Error processing chat query: {str(e)}")
        raise e
