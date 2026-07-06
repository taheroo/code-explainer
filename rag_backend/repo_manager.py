from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

CLONE_DIR = Path(__file__).resolve().parent.parent / "cloned_repos"
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", "eval", "rag_backend"}


def sync_and_get_commit(repo_path: Path) -> str:
    """Pull latest, then return current HEAD commit hash."""
    subprocess.run(["git", "pull", "--ff-only"], cwd=repo_path, capture_output=True, text=True)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_path, text=True).strip()


def read_last_ingested_commit(repo_name: str) -> str | None:
    from qdrant_wrapper import get_qdrant_client
    from qdrant_client.http.models import FieldCondition, MatchValue, Filter
    client = get_qdrant_client()
    try:
        result = client._client.scroll(
            collection_name=client.settings.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="repo_name", match=MatchValue(value=repo_name)),
                    FieldCondition(key="_meta", match=MatchValue(value="commit_hash")),
                ]
            ),
            limit=1,
            with_payload=True,
        )
        points = result[0]
        if points:
            return points[0].payload.get("value")
    except Exception:
        pass
    return None


def write_last_ingested_commit(repo_name: str, commit_hash: str):
    import uuid
    from qdrant_wrapper import get_qdrant_client
    from qdrant_client.http.models import PointStruct
    client = get_qdrant_client()
    client._client.upsert(
        collection_name=client.settings.collection_name,
        points=[
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{repo_name}_commit")),
                vector={"dense": [0.0] * client.settings.vector_size},
                payload={
                    "repo_name": repo_name,
                    "_meta": "commit_hash",
                    "value": commit_hash,
                },
            )
        ],
    )


def _git_clone(url: str, dest: Path, token: Optional[str] = None) -> None:
    if dest.exists():
        log.info("Already exists, skipping clone: %s", dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if token:
        url = url.replace("https://", f"https://{token}@")
    log.info("Cloning %s into %s", url, dest)
    subprocess.run(["git", "clone", "--depth=1", url, str(dest)], check=True)


def clone_single_repo(url: str, token: Optional[str] = None) -> tuple[str, Path]:
    name = Path(url.rstrip("/").rstrip(".git")).name
    dest = CLONE_DIR / name
    _git_clone(url, dest, token)
    repos = []
    for child in sorted(dest.iterdir()):
        if child.is_dir() and child.name not in SKIP_DIRS:
            repos.append((child.name, child))
    root_files = [f for f in dest.iterdir() if f.is_file()]
    if root_files:
        repos.insert(0, ("_root", dest))
    if not repos:
        repos.append((name, dest))
    return (name, dest)


def resolve_repos() -> list[tuple[str, Path]]:
    mode = os.getenv("REPO_MODE", "monorepo")
    token = os.getenv("GITHUB_TOKEN") or None

    if mode == "list":
        raw = os.getenv("REPOS", "[]")
        entries: list[dict] = json.loads(raw)
        repos: list[tuple[str, Path]] = []
        for entry in entries:
            name = entry["name"]
            dest = CLONE_DIR / name
            _git_clone(entry["url"], dest, token)
            repos.append((name, dest))
        return repos

    monorepo_url = os.getenv("MONOREPO_URL", "")
    if monorepo_url:
        _git_clone(monorepo_url, CLONE_DIR, token)

    if not CLONE_DIR.exists():
        return []

    repos = []
    for child in sorted(CLONE_DIR.iterdir()):
        if child.is_dir() and child.name not in SKIP_DIRS:
            repos.append((child.name, child))

    root_files = [f for f in CLONE_DIR.iterdir() if f.is_file()]
    if root_files:
        repos.insert(0, ("_root", CLONE_DIR))

    return repos
