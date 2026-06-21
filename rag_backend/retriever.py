from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

from embedder import embed_query
from qdrant_client import get_qdrant_client
from sparse_embedder import embed_sparse

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
    parent_text: str = ""
    parent_start_line: int = 0
    parent_end_line: int = 0

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


def _search_variant(
    query: str,
    client: Any,
    top_k: int,
    target_repo: str | None,
    metadata_filter: dict | None,
) -> list[RetrievedChunk]:
    vector = embed_query(query)
    sparse = embed_sparse(query)
    hits = client.search(
        vector=vector,
        sparse_vector=sparse,
        limit=top_k * 4,
        repo_name=target_repo,
        metadata_filter=metadata_filter,
    )
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
                parent_text=str(payload.get("parent_text", "")),
                parent_start_line=int(payload.get("parent_start_line", 0) or 0),
                parent_end_line=int(payload.get("parent_end_line", 0) or 0),
            )
        )
    return results


def _rrf_merge(
    per_variant_results: list[list[RetrievedChunk]],
    k: int = 60,
) -> list[RetrievedChunk]:
    key_score: dict[str, dict] = {}
    for vid, variant_results in enumerate(per_variant_results):
        for rank, chunk in enumerate(variant_results):
            key = f"{chunk.file_path}::{chunk.start_line}::{chunk.end_line}"
            if key not in key_score:
                key_score[key] = {"chunk": chunk, "rrf": 0.0}
            key_score[key]["rrf"] += 1.0 / (k + rank + 1)

    scored = sorted(key_score.values(), key=lambda x: x["rrf"], reverse=True)
    for entry in scored:
        entry["chunk"].score = entry["rrf"]
    return [entry["chunk"] for entry in scored]


def retrieve(question: str, target_repo: str | None = None, top_k: int = 5) -> list[RetrievedChunk]:
    from llm import expand_query, parse_query

    client = get_qdrant_client()
    client.ensure_collection()

    cleaned_query, metadata_filter = parse_query(question)
    variants = expand_query(cleaned_query)

    per_variant_results: list[list[RetrievedChunk]] = []
    for variant in variants:
        results = _search_variant(variant, client, top_k, target_repo, metadata_filter)
        per_variant_results.append(results)

    merged = _rrf_merge(per_variant_results)
    seen: dict[str, RetrievedChunk] = {}
    for c in merged:
        if c.file_path not in seen or c.score > seen[c.file_path].score:
            seen[c.file_path] = c
    deduped = [seen[c.file_path] for c in merged if c.file_path in seen and seen[c.file_path] is c]

    reranked = _rerank(question, deduped, top_k=top_k)
    _normalize_scores(reranked)
    if reranked and reranked[0].score < CONFIDENCE_THRESHOLD:
        return []
    return reranked
