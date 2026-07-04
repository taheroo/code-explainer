from __future__ import annotations

import argparse
import gc
import hashlib
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chunker import chunk_code
from embedder import embed_texts
from qdrant_wrapper import get_qdrant_client
from sparse_embedder import embed_sparse_batch

log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".py", ".js", ".jsx", ".ts", ".tsx", ".md", ".sql"}
SKIP_DIRS = {"node_modules", ".git", "__pycache__", "dist", "build", ".venv", "venv"}

MAX_PARENT_CHARS = 50000




# ----------------------------
# DATA MODEL
# ----------------------------

@dataclass(frozen=True)
class ChunkRecord:
    repo_name: str
    file_path: str
    chunk_index: int
    text: str
    start_line: int
    end_line: int
    symbol_name: str
    language: str
    parent_text: str = ""
    parent_start_line: int = 0
    parent_end_line: int = 0

    @property
    def id(self) -> str:
        key = f"{self.repo_name}::{self.file_path}::{self.chunk_index}::{self.text}"
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return str(uuid.UUID(h[:32]))

    @property
    def payload(self) -> dict[str, Any]:
        return {
            "repo_name": self.repo_name,
            "file_path": self.file_path,
            "chunk_index": self.chunk_index,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "symbol_name": self.symbol_name,
            "language": self.language,
            "text": self.text,
            "char_count": len(self.text),
            "parent_text": self.parent_text,
            "parent_start_line": self.parent_start_line,
            "parent_end_line": self.parent_end_line,
        }


def get_repo_paths() -> list[tuple[str, Path]]:
    from repo_manager import resolve_repos
    return resolve_repos()


# ----------------------------
# FILE DISCOVERY
# ----------------------------

def iter_repo_files(repo_path: Path, recurse: bool = True):
    files = repo_path.rglob("*") if recurse else repo_path.iterdir()
    for path in files:
        if not path.is_file():
            continue

        if any(part in SKIP_DIRS for part in path.parts):
            continue

        if path.stem.startswith("test_"):
            continue

        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


# ----------------------------
# CHUNKING
# ----------------------------

def collect_chunks_from_repo(repo_name: str, repo_path: Path) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []

    log.info("🔍 Scanning repo directory: %s", repo_path)

    recurse = repo_name != "_root"
    for file_path in iter_repo_files(repo_path, recurse=recurse):
        try:
            source = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            log.warning("Skipping file %s: %s", file_path, exc)
            continue

        raw_chunks = chunk_code(file_path.as_posix(), source)

        # 🔥 HARD SAFETY: NEVER allow empty ingestion silently
        if not raw_chunks:
            log.warning("⚠️ Chunker empty → fallback raw file used: %s", file_path)
            raw_chunks = [{
                "text": source,
                "start": 0,
                "end": len(source.splitlines()),
                "symbol": file_path.name,
            }]

        rel_path = file_path.relative_to(repo_path).as_posix()
        language = file_path.suffix.lower().lstrip(".") or "unknown"

        total_lines = len(source.splitlines())
        parent_text = source[:MAX_PARENT_CHARS] if len(source) > MAX_PARENT_CHARS else source

        for idx, item in enumerate(raw_chunks):
            text = str(item.get("text", "")).strip()
            if not text:
                continue

            chunks.append(
                ChunkRecord(
                    repo_name=repo_name,
                    file_path=rel_path,
                    chunk_index=idx,
                    text=text,
                    start_line=int(item.get("start", 0)),
                    end_line=int(item.get("end", 0)),
                    symbol_name=str(item.get("symbol", "")),
                    language=language,
                    parent_text=parent_text,
                    parent_start_line=1,
                    parent_end_line=total_lines,
                )
            )

    log.info("📦 Repo %s → %d chunks created", repo_name, len(chunks))
    return chunks


# ----------------------------
# VECTOR BUILD
# ----------------------------

def build_points(chunks: list[ChunkRecord]) -> list[dict[str, Any]]:
    log.info("Generating dense embeddings for %d chunks...", len(chunks))
    vectors = []
    batch_size = 16
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        batch_vectors = embed_texts(chunk.text for chunk in batch)
        vectors.extend(batch_vectors)
        log.info("Encoded %d/%d chunks", min(i + batch_size, len(chunks)), len(chunks))
        gc.collect()

    log.info("Generating sparse embeddings...")
    sparse_vectors = embed_sparse_batch(chunk.text for chunk in chunks)

    points = []
    for chunk, vector, sparse in zip(chunks, vectors, sparse_vectors):
        points.append({
            "id": chunk.id,
            "vector": vector,
            "sparse": {"indices": [i for i, _ in sparse], "values": [v for _, v in sparse]} if sparse else None,
            "payload": chunk.payload,
        })

    return points


# ----------------------------
# INGEST SINGLE REPO
# ----------------------------

def ingest_repo(repo_name: str, repo_path: Path, dry_run: bool = False) -> int:
    client = get_qdrant_client()
    client.ensure_collection()

    chunks = collect_chunks_from_repo(repo_name, repo_path)

    if not chunks:
        log.warning("❌ No chunks found in repo: %s", repo_name)
        return 0

    if dry_run:
        return len(chunks)

    client.delete_repo(repo_name)

    log.info("Embedding %d chunks from %s...", len(chunks), repo_name)
    points = build_points(chunks)
    log.info("Built %d points — upserting to Qdrant...", len(points))

    batch_size = 64
    for i in range(0, len(points), batch_size):
        client.upsert_points(points[i:i + batch_size])
        if (i // batch_size) % 5 == 0:
            log.info("Upserted %d/%d points", min(i + batch_size, len(points)), len(points))

    log.info("✅ Indexed %d chunks from %s", len(chunks), repo_name)
    return len(chunks)


# ----------------------------
# INGEST ALL
# ----------------------------

def ingest_all(target_repo: str | None = None, dry_run: bool = False) -> int:
    repo_paths = get_repo_paths()

    if target_repo:
        repo_paths = [(n, p) for n, p in repo_paths if n == target_repo]
        if not repo_paths:
            raise ValueError(f"Unknown repo: {target_repo}")

    total = 0

    for repo_name, repo_path in repo_paths:
        if not repo_path.exists():
            log.warning("⚠️ Missing repo folder: %s", repo_path)
            continue

        total += ingest_repo(repo_name, repo_path, dry_run=dry_run)

    log.info("🚀 TOTAL chunks indexed: %d", total)
    return total


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    ingest_all(target_repo=args.repo, dry_run=args.dry_run)


if __name__ == "__main__":
    main()