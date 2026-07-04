from __future__ import annotations

from functools import lru_cache
from typing import Iterable

from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedder(model_name: str = MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_texts(texts: Iterable[str]) -> list[list[float]]:
    items = list(texts)
    if not items:
        return []

    model = get_embedder()
    vectors = model.encode(items, normalize_embeddings=True, show_progress_bar=False)
    if hasattr(vectors, "tolist"):
        return vectors.tolist()
    return [vector.tolist() for vector in vectors]


def embed_query(text: str) -> list[float]:
    model = get_embedder()
    vector = model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
    return vector.tolist()
