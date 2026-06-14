from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

from .embedder import embed_query
from .qdrant_client import get_qdrant_client
from .sparse_embedder import embed_sparse

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
CONFIDENCE_THRESHOLD = -999.0


class QueryRequest(BaseModel):
    question: str
    target_repo: Optional[str] = None
    top_k: int = 5
    session_id: Optional[str] = None


class RetrievedChunk(BaseModel):
    text: str
    repo_name: str
    file_path: str
    symbol_name: str = ""
    language: str = ""
    start_line: int = 0
    end_line: int = 0
    score: float = 0.0
    confidence: float = 0.0

    @property
    def source(self) -> dict[str, Any]:
        conf_pct = round(self.confidence * 100, 1)
        label = "High" if conf_pct > 80 else "Medium" if conf_pct >= 50 else "Low"
        return {
            "folder_name": self.file_path.split("/")[0],
            "file_path": self.file_path,
            "symbol_name": self.symbol_name,
            "language": self.language,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "confidence": conf_pct,
            "confidence_label": label,
        }


@lru_cache(maxsize=1)
def _get_cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(CROSS_ENCODER_MODEL)


def _rerank(question: str, chunks: list[RetrievedChunk], top_k: int = 5) -> list[RetrievedChunk]:
    if not chunks:
        return chunks
    pairs = [(question, c.text) for c in chunks]
    scores = _get_cross_encoder().predict(pairs, show_progress_bar=False)
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    for c, s in ranked:
        c.score = float(s)
    return [c for c, _ in ranked[:top_k]]


def _normalize_scores(chunks: list[RetrievedChunk]) -> None:
    if not chunks:
        return
    max_score = max(c.score for c in chunks)
    if max_score <= 0:
        for c in chunks:
            c.confidence = 0.0
        return
    for c in chunks:
        raw = c.score / max_score
        c.confidence = round(max(0.0, min(raw, 1.0)), 4)


def _deduplicate_by_path(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    seen: dict[str, RetrievedChunk] = {}
    for c in chunks:
        if c.file_path not in seen or c.score > seen[c.file_path].score:
            seen[c.file_path] = c
    return [seen[c.file_path] for c in chunks if c.file_path in seen and seen[c.file_path] is c]


def retrieve(question: str, target_repo: str | None = None, top_k: int = 5) -> list[RetrievedChunk]:
    client = get_qdrant_client()
    client.ensure_collection()
    vector = embed_query(question)
    sparse = embed_sparse(question)
    hits = client.search(vector=vector, sparse_vector=sparse, limit=top_k * 4, repo_name=target_repo)

    results: list[RetrievedChunk] = []
    for hit in hits:
        payload = hit.get("payload", {})
        results.append(
            RetrievedChunk(
                text=str(payload.get("text", "")),
                repo_name=str(payload.get("repo_name", "unknown")),
                file_path=Path(str(payload.get("file_path", "unknown"))).as_posix(),
                symbol_name=str(payload.get("symbol_name", "")),
                language=str(payload.get("language", "")),
                start_line=int(payload.get("start_line", 0) or 0),
                end_line=int(payload.get("end_line", 0) or 0),
                score=float(hit.get("score", 0.0)),
            )
        )

    reranked = _deduplicate_by_path(_rerank(question, results, top_k=top_k))
    _normalize_scores(reranked)
    if reranked and reranked[0].score < CONFIDENCE_THRESHOLD:
        return []
    return reranked
