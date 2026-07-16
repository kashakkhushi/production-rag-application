import chromadb
from chromadb.config import Settings
from app.core.config import settings
from typing import List, Dict, Any

class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH, settings=Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(
            name="rag_collection",
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        ids = [chunk["chunk_id"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = []
        for chunk in chunks:
            # Metadata values must be int, float, str or bool
            meta = {
                "document_id": chunk["document_id"],
                "filename": chunk["filename"],
                "page_number": int(chunk["page_number"]),
                "chunk_id": chunk["chunk_id"],
                "chunk_index": int(chunk["chunk_index"])
            }
            metadatas.append(meta)
            
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )

    def delete_by_document_id(self, document_id: str):
        self.collection.delete(
            where={"document_id": document_id}
        )

    def search(self, query_embedding: List[float], top_k: int) -> Dict[str, Any]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        return results

vector_store = VectorStore()
