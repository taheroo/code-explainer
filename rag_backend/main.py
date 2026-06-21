from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
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


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Code Explorer Chat</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; display: flex; justify-content: center; padding: 40px 16px; }
    .container { max-width: 720px; width: 100%; background: #fff; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); display: flex; flex-direction: column; height: 80vh; }
    h1 { font-size: 1.2rem; padding: 16px 20px; border-bottom: 1px solid #eee; color: #333; }
    #chat { flex: 1; overflow-y: auto; padding: 16px 20px; display: flex; flex-direction: column; gap: 12px; }
    .msg { max-width: 85%; padding: 10px 14px; border-radius: 10px; line-height: 1.5; font-size: 0.95rem; }
    .msg.user { background: #007aff; color: #fff; align-self: flex-end; }
    .msg.bot { background: #f0f0f0; color: #222; align-self: flex-start; }
    .msg.bot .answer { white-space: pre-wrap; }
    .msg.bot .answer h2 { font-size: 1rem; margin: 8px 0 4px; }
    .msg.bot .answer hr { border: none; border-top: 1px solid #ddd; margin: 8px 0; }
    .msg.bot .source { font-size: 0.8rem; color: #666; margin-top: 4px; }
    .input-row { display: flex; gap: 8px; padding: 12px 20px; border-top: 1px solid #eee; }
    .input-row input { flex: 1; padding: 10px 14px; border: 1px solid #ddd; border-radius: 8px; font-size: 0.95rem; outline: none; }
    .input-row input:focus { border-color: #007aff; }
    .input-row button { padding: 10px 20px; background: #007aff; color: #fff; border: none; border-radius: 8px; font-size: 0.95rem; cursor: pointer; }
    .input-row button:hover { background: #005bbf; }
    .typing { color: #999; font-style: italic; font-size: 0.85rem; padding: 4px 14px; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Ask about your codebase</h1>
    <div id="chat">
      <div class="msg bot"><div class="answer">Ask me anything about your code. I'll search the codebase and give you an answer grounded in the actual source.</div></div>
    </div>
    <div id="typing" class="typing" style="display:none; padding: 0 20px 4px;">Thinking...</div>
    <div class="input-row">
      <input id="q" type="text" placeholder="Type your question..." autofocus />
      <button onclick="ask()">Send</button>
    </div>
  </div>
  <script>
    const chat = document.getElementById('chat');
    const typing = document.getElementById('typing');
    const input = document.getElementById('q');

    function escapeHTML(str) {
      return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }

    function mdToHTML(md) {
      return md
        .replace(/### (.+)/g, '<h3>$1</h3>')
        .replace(/## (.+)/g, '<h2>$1</h2>')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n/g, '<br>');
    }

    function addMsg(text, isUser) {
      const d = document.createElement('div');
      d.className = 'msg ' + (isUser ? 'user' : 'bot');
      d.innerHTML = isUser ? escapeHTML(text) : '<div class="answer">' + mdToHTML(escapeHTML(text)) + '</div>';
      chat.appendChild(d);
      chat.scrollTop = chat.scrollHeight;
    }

    async function ask() {
      const q = input.value.trim();
      if (!q) return;
      addMsg(q, true);
      input.value = '';
      typing.style.display = 'block';
      try {
        const r = await fetch('/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: q })
        });
        const d = await r.json();
        typing.style.display = 'none';
        addMsg(d.answer, false);
      } catch(e) {
        typing.style.display = 'none';
        addMsg('Error: ' + e.message, false);
      }
    }

    input.addEventListener('keydown', e => { if (e.key === 'Enter') ask(); });
  </script>
</body>
</html>"""


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
