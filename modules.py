from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from typing import List

class E5Embedder:
    def __init__(self):
        self.model = SentenceTransformer("models/e5-small-v2")
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).tolist()
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()

class MsMarcoCrossEncoder:
    def __init__(self):
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


