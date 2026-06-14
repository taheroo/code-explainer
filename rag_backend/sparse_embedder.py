from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "and", "but", "or", "if", "while", "that", "this", "it", "its",
    "we", "you", "they", "he", "she", "they", "them", "their", "our",
})


@lru_cache(maxsize=1)
def _get_vocab() -> dict[str, int]:
    return {}


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r"[a-zA-Z_]\w*", text)
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


def embed_sparse(text: str) -> list[tuple[int, float]]:
    from collections import defaultdict

    tokens = _tokenize(text)
    freq: dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    if not freq:
        return []
    max_freq = max(freq.values())
    merged: dict[int, float] = defaultdict(float)
    for t, f in freq.items():
        idx = abs(hash(t)) % (10 ** 6)
        val = f / max_freq
        merged[idx] = max(merged[idx], val)
    return list(zip(merged.keys(), merged.values()))


def embed_sparse_batch(texts: Iterable[str]) -> list[list[tuple[int, float]]]:
    return [embed_sparse(t) for t in texts]
