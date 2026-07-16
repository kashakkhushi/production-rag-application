from sentence_transformers import CrossEncoder
from typing import List, Dict, Any

class RerankerService:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not chunks:
            return []
            
        pairs = [[query, chunk["text"]] for chunk in chunks]
        scores = self.model.predict(pairs)
        
        for idx, score in enumerate(scores):
            chunks[idx]["reranker_score"] = float(score)
            
        # Sort by reranker score descending
        chunks.sort(key=lambda x: x["reranker_score"], reverse=True)
        return chunks[:top_k]

reranker_service = RerankerService()
