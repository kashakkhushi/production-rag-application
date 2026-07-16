from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, Optional
import asyncio
from app.rag.vector_store import vector_store
from loguru import logger
import re

def tokenize(text: str) -> List[str]:
    # Simple whitespace and punctuation tokenizer
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

class SparseIndexManager:
    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_chunks: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
        
    async def rebuild_index(self):
        async with self._lock:
            try:
                logger.info("Rebuilding BM25 index from ChromaDB...")
                # Fetch all data from Chroma
                result = vector_store.collection.get(include=["documents", "metadatas"])
                
                documents = result.get("documents", [])
                metadatas = result.get("metadatas", [])
                ids = result.get("ids", [])
                
                if not documents:
                    self.bm25 = None
                    self.corpus_chunks = []
                    logger.info("No documents found. BM25 index is empty.")
                    return
                    
                new_corpus_chunks = []
                tokenized_corpus = []
                
                for doc, meta, chunk_id in zip(documents, metadatas, ids):
                    chunk_data = {
                        "chunk_id": chunk_id,
                        "document_id": meta["document_id"],
                        "filename": meta["filename"],
                        "page_number": meta["page_number"],
                        "chunk_index": meta["chunk_index"],
                        "text": doc
                    }
                    new_corpus_chunks.append(chunk_data)
                    tokenized_corpus.append(tokenize(doc))
                    
                new_bm25 = BM25Okapi(tokenized_corpus)
                
                # Atomic assignment
                self.corpus_chunks = new_corpus_chunks
                self.bm25 = new_bm25
                logger.info(f"Successfully rebuilt BM25 index with {len(documents)} chunks.")
            except Exception as e:
                logger.error(f"Failed to rebuild BM25 index: {str(e)}")

    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if not self.bm25 or not self.corpus_chunks:
            return []
            
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                chunk_data = self.corpus_chunks[idx].copy()
                chunk_data["score"] = scores[idx]
                results.append(chunk_data)
                
        return results

sparse_index = SparseIndexManager()
