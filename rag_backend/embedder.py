from __future__ import annotations

from functools import lru_cache
from typing import Iterable

from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"


@lru_cache(maxsize=1)
def get_embedder(model_name: str = MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_texts(texts: Iterable[str]) -> list[list[float]]:
    items = [f"passage: {text}" for text in texts]
    if not items:
        return []

    model = get_embedder()
    batch_size = 32
    all_embeddings: list[list[float]] = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        vectors = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        if hasattr(vectors, "tolist"):
            all_embeddings.extend(vectors.tolist())
        else:
            all_embeddings.extend(vector.tolist() for vector in vectors)
    return all_embeddings


def embed_query(text: str) -> list[float]:
    model = get_embedder()
    vector = model.encode([f"query: {text}"], normalize_embeddings=True, show_progress_bar=False)[0]
    return vector.tolist()

