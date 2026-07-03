from __future__ import annotations

from functools import lru_cache
from typing import Iterable

from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"


@lru_cache(maxsize=1)
def get_embedder(model_name: str = MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_texts(texts: Iterable[str]) -> list[list[float]]:
    items = list(texts)
    if not items:
        return []

    model = get_embedder()
    all_embeddings = []
    for i in range(0, len(items), 32):
        batch = [f"passage: {t}" for t in items[i:i+32]]
        embeddings = model.encode(batch, normalize_embeddings=True)
        all_embeddings.extend(embeddings.tolist())
    return all_embeddings


def embed_query(text: str) -> list[float]:
    model = get_embedder()
    vector = model.encode([f"query: {text}"], normalize_embeddings=True, show_progress_bar=False)[0]
    return vector.tolist()

