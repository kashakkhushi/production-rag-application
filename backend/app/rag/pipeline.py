import time
from typing import List, Dict, Any, Tuple
from app.core.config import settings
from app.rag.embeddings import embedding_service
from app.rag.vector_store import vector_store
from app.rag.sparse_index import sparse_index
from app.rag.reranker import reranker_service
from loguru import logger

def min_max_normalize(scores: List[float]) -> List[float]:
    """Applies Min-Max normalization to a list of scores.

    Normalizes the scores to a range of [0.0, 1.0]. If all scores are identical,
    they are mapped to 0.5 to represent neutral confidence.

    Args:
        scores (List[float]): A list of raw scores from a retrieval system.

    Returns:
        List[float]: A list of normalized scores in the range [0.0, 1.0].
    """
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [0.5 for _ in scores]
    return [(s - min_score) / (max_score - min_score) for s in scores]

def run_hybrid_retrieval(query: str, top_k: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Executes the complete hybrid retrieval and reranking pipeline.

    This function performs dual retrieval using dense embeddings (ChromaDB) and
    sparse lexical matching (BM25). The results are min-max normalized, combined
    using configurable weights (DENSE_WEIGHT and SPARSE_WEIGHT), deduplicated, 
    and finally reranked using a Cross-Encoder model.

    Args:
        query (str): The user's input question.
        top_k (int): The number of top chunks to retrieve from each index initially.

    Returns:
        Tuple[List[Dict[str, Any]], Dict[str, Any]]: 
            - A list of the top reranked document chunks (dict).
            - A dictionary containing retrieval latency and score statistics.
    """
    start_time = time.time()
    stats = {
        "dense_scores": [],
        "sparse_scores": [],
        "reranker_scores": [],
        "retrieval_latency_ms": 0.0,
        "retrieved_chunks_count": 0
    }
    
    # 1. Dense Retrieval
    query_emb = embedding_service.encode_query(query)
    dense_results = vector_store.search(query_emb, top_k=top_k)
    
    dense_candidates = {}
    if dense_results and "distances" in dense_results and dense_results["distances"]:
        # distances are typically cosine distance (smaller is better). Convert to similarity.
        distances = dense_results["distances"][0]
        ids = dense_results["ids"][0]
        docs = dense_results["documents"][0]
        metas = dense_results["metadatas"][0]
        
        # Convert distance to similarity for normalization
        similarities = [1.0 - d for d in distances]
        norm_dense_scores = min_max_normalize(similarities)
        stats["dense_scores"] = similarities
        
        for idx, chunk_id in enumerate(ids):
            dense_candidates[chunk_id] = {
                "chunk_id": chunk_id,
                "text": docs[idx],
                "document_id": metas[idx]["document_id"],
                "filename": metas[idx]["filename"],
                "page_number": metas[idx]["page_number"],
                "chunk_index": metas[idx]["chunk_index"],
                "dense_score": norm_dense_scores[idx],
                "sparse_score": 0.0
            }
            
    # 2. Sparse Retrieval
    sparse_results = sparse_index.search(query, top_k=top_k)
    if sparse_results:
        raw_sparse = [res["score"] for res in sparse_results]
        stats["sparse_scores"] = raw_sparse
        norm_sparse_scores = min_max_normalize(raw_sparse)
        
        for idx, res in enumerate(sparse_results):
            chunk_id = res["chunk_id"]
            if chunk_id in dense_candidates:
                dense_candidates[chunk_id]["sparse_score"] = norm_sparse_scores[idx]
            else:
                dense_candidates[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": res["text"],
                    "document_id": res["document_id"],
                    "filename": res["filename"],
                    "page_number": res["page_number"],
                    "chunk_index": res["chunk_index"],
                    "dense_score": 0.0,
                    "sparse_score": norm_sparse_scores[idx]
                }
                
    # 3. Combine scores
    combined_candidates = []
    for chunk in dense_candidates.values():
        chunk["combined_score"] = (settings.DENSE_WEIGHT * chunk["dense_score"]) + (settings.SPARSE_WEIGHT * chunk["sparse_score"])
        combined_candidates.append(chunk)
        
    combined_candidates.sort(key=lambda x: x["combined_score"], reverse=True)
    
    # 4. Rerank
    # Rerank the top 15 candidates
    candidates_to_rerank = combined_candidates[:15]
    final_results = reranker_service.rerank(query, candidates_to_rerank, settings.RERANK_TOP_K)
    
    stats["reranker_scores"] = [res.get("reranker_score", 0.0) for res in final_results]
    stats["retrieved_chunks_count"] = len(final_results)
    stats["retrieval_latency_ms"] = (time.time() - start_time) * 1000
    
    return final_results, stats
