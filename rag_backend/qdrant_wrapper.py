from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

from qdrant_client import QdrantClient as _QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

_QDRANT_INSTANCE: LocalQdrantClient | None = None


@dataclass(frozen=True)
class QdrantSettings:
    collection_name: str = os.getenv("QDRANT_COLLECTION", "codebase")
    vector_size: int = int(os.getenv("EMBEDDING_DIM", "384"))


def _find_qdrant_storage() -> str:
    storage_path = os.getenv("QDRANT_STORAGE_PATH")
    if storage_path:
        return storage_path
    return str(Path(__file__).resolve().parent.parent / "qdrant_storage")


def _clean_stale_lock(storage_path: str) -> None:
    lock_file = Path(storage_path) / ".lock"
    try:
        import portalocker
        portalocker.Lock(str(lock_file), flags=portalocker.LOCK_EX).release()
    except Exception:
        pass
    if lock_file.exists():
        try:
            lock_file.unlink()
        except OSError:
            pass


def _create_local_client(storage_path: str, max_retries: int = 5, delay: float = 1.0) -> _QdrantClient:
    for attempt in range(max_retries):
        try:
            return _QdrantClient(path=storage_path)
        except RuntimeError as exc:
            if "already accessed" in str(exc) and attempt < max_retries - 1:
                _clean_stale_lock(storage_path)
                time.sleep(delay * (attempt + 1))
                continue
            raise


class LocalQdrantClient:
    def __init__(self, settings: QdrantSettings | None = None):
        self.settings = settings or QdrantSettings()
        url = os.getenv("QDRANT_URL")
        if url:
            self._client = _QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY"))
        else:
            host = os.getenv("QDRANT_HOST")
            if host:
                port = int(os.getenv("QDRANT_PORT", "6333"))
                self._client = _QdrantClient(host=host, port=port)
            else:
                storage_path = _find_qdrant_storage()
                _clean_stale_lock(storage_path)
                self._client = _create_local_client(storage_path)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> LocalQdrantClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def ensure_collection(self) -> None:
        names = [c.name for c in self._client.get_collections().collections]
        if self.settings.collection_name in names:
            return
        self._client.create_collection(
            collection_name=self.settings.collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=self.settings.vector_size,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(),
            },
        )

    def upsert_points(self, points: list[dict[str, Any]]) -> None:
        if not points:
            return
        point_structs = []
        for p in points:
            pid = p.get("id", "")
            payload = p.get("payload", {})
            vector = p.get("vector")
            sparse_data = p.get("sparse")
            if vector is None:
                continue
            try:
                uid = str(uuid.UUID(pid[:32]))
            except (ValueError, TypeError):
                uid = str(uuid.uuid4())
            if sparse_data:
                vec = {
                    "dense": vector,
                    "sparse": SparseVector(
                        indices=sparse_data["indices"],
                        values=sparse_data["values"],
                    ),
                }
            else:
                vec = vector
            point_structs.append(PointStruct(id=uid, vector=vec, payload=payload))
        self._client.upsert(
            collection_name=self.settings.collection_name,
            points=point_structs,
        )

    def delete_repo(self, repo_name: str) -> None:
        try:
            self._client.delete(
                collection_name=self.settings.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="repo_name",
                            match=MatchValue(value=repo_name),
                    )
                ]
            ),
        )
        except Exception:
            log.warning("Failed to delete existing data for repo '%s' (index may not exist yet)", repo_name)

    def search(
        self,
        vector: list[float],
        sparse_vector: list[tuple[int, float]] | None = None,
        limit: int = 5,
        repo_name: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        conditions: list[FieldCondition] = []
        if repo_name:
            conditions.append(
                FieldCondition(key="repo_name", match=MatchValue(value=repo_name))
            )
        if metadata_filter:
            for key, value in metadata_filter.items():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
        qfilter = Filter(must=conditions) if conditions else None
        dense_limit = limit * 4
        dense_result = self._client.query_points(
            collection_name=self.settings.collection_name,
            query=vector,
            using="dense",
            query_filter=qfilter,
            limit=dense_limit,
            with_payload=True,
            with_vectors=False,
        )
        dense_hits = {(str(pt.id), pt.payload.get("repo_name", ""), pt.payload.get("file_path", "")): (i, pt) for i, pt in enumerate(dense_result.points)}

        if sparse_vector:
            sparse_result = self._client.query_points(
                collection_name=self.settings.collection_name,
                query=SparseVector(indices=[i for i, _ in sparse_vector], values=[v for _, v in sparse_vector]),
                using="sparse",
                query_filter=qfilter,
                limit=dense_limit,
                with_payload=True,
                with_vectors=False,
            )
            sparse_hits = {(str(pt.id), pt.payload.get("repo_name", ""), pt.payload.get("file_path", "")): (i, pt) for i, pt in enumerate(sparse_result.points)}
            all_keys = set(dense_hits.keys()) | set(sparse_hits.keys())
            K = 60
            scored = []
            for key in all_keys:
                dense_rank, dense_pt = dense_hits.get(key, (None, None))
                sparse_rank, sparse_pt = sparse_hits.get(key, (None, None))
                rrf = 0.0
                if dense_pt is not None:
                    rrf += 1.0 / (K + dense_rank + 1)
                if sparse_pt is not None:
                    rrf += 1.0 / (K + sparse_rank + 1)
                pt = dense_pt or sparse_pt
                scored.append((rrf, pt))
            scored.sort(key=lambda x: x[0], reverse=True)
            merged = scored[:limit]
        else:
            merged = [(pt.score, pt) for pt in dense_result.points][:limit]

        return [
            {
                "id": str(pt.id),
                "score": float(score),
                "payload": pt.payload or {},
            }
            for score, pt in merged
        ]


def get_qdrant_client() -> LocalQdrantClient:
    global _QDRANT_INSTANCE
    if _QDRANT_INSTANCE is None:
        _QDRANT_INSTANCE = LocalQdrantClient()
    return _QDRANT_INSTANCE
