from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingService:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def encode_query(self, query: str) -> List[float]:
        # BGE models sometimes use "Represent this sentence for searching relevant passages: " for queries
        # For simplicity and standard behavior, we encode as is or append the prefix if needed
        # We will just encode directly as it usually performs well, but appending instruction helps for BGE.
        # "Represent this sentence for searching relevant passages: "
        # We'll use the plain query here to keep it agnostic, but BGE recommends instructions for queries.
        q = f"Represent this sentence for searching relevant passages: {query}"
        embedding = self.model.encode([q], normalize_embeddings=True)[0]
        return embedding.tolist()

embedding_service = EmbeddingService()
