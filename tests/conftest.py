import numpy as np


class MockEmbeddings:
    """Deterministic embedding model for unit tests (no Ollama required)."""

    def __init__(self, dim: int = 8):
        self.dim = dim

    def embed_documents(self, docs):
        return [self._vector(text) for text in docs]

    def embed_query(self, query):
        return self._vector(query)

    def _vector(self, text: str):
        seed = sum(ord(c) for c in text) % 97
        rng = np.random.default_rng(seed)
        return rng.random(self.dim).astype(np.float32).tolist()
