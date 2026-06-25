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


def _git_clone(url: str, dest: Path, token: Optional[str] = None) -> None:
    if dest.exists():
        log.info("Already exists, skipping clone: %s", dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if token:
        url = url.replace("https://", f"https://{token}@")
    log.info("Cloning %s into %s", url, dest)
    subprocess.run(["git", "clone", "--depth=1", url, str(dest)], check=True)


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
    if not monorepo_url:
        raise ValueError("MONOREPO_URL is required when REPO_MODE=monorepo")
    _git_clone(monorepo_url, CLONE_DIR, token)

    repos = []
    for child in sorted(CLONE_DIR.iterdir()):
        if child.is_dir() and child.name not in SKIP_DIRS:
            repos.append((child.name, child))

    root_files = [f for f in CLONE_DIR.iterdir() if f.is_file()]
    if root_files:
        repos.insert(0, ("_root", CLONE_DIR))

    return repos
