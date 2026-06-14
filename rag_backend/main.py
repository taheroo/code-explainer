from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parent / ".env")

from .ingest import ingest_all, ingest_repo
from .llm import generate_answer
from .retriever import QueryRequest, retrieve
from .repo_manager import resolve_repos


@asynccontextmanager
async def lifespan(app: FastAPI):
    repos = resolve_repos()
    for repo_name, repo_path in repos:
        ingest_repo(repo_name=repo_name, repo_path=repo_path)
    yield


app = FastAPI(title="Rendoo RAG Backend", lifespan=lifespan)

session_history: dict[str, list[dict]] = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class IngestRequest(BaseModel):
    repo: Optional[str] = None
    dry_run: bool = False


class QueryResult(BaseModel):
    answer: str
    sources: list[dict]


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "Rendoo RAG Backend", "status": "running", "docs": "/docs"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest")
def ingest(request: IngestRequest) -> dict[str, int | str]:
    try:
        total = ingest_all(target_repo=request.repo, dry_run=request.dry_run)
        return {"status": "ok", "chunks_indexed": total}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/query")
def query(request: QueryRequest) -> QueryResult:
    try:
        chunks = retrieve(request.question, target_repo=request.target_repo, top_k=request.top_k)
        answer = generate_answer(request.question, chunks)
        return QueryResult(
            answer=answer,
            sources=[chunk.source for chunk in chunks],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/session/{session_id}/clear")
def clear_session(session_id: str) -> dict[str, str]:
    session_history.pop(session_id, None)
    return {"status": "ok", "session_id": session_id}
