from __future__ import annotations

import hashlib
import logging
import time
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parent / ".env")

log = logging.getLogger(__name__)

from embedder import get_embedder
from qdrant_wrapper import get_qdrant_client
from ingest import ingest_all, ingest_repo
from llm import stream_answer
from retriever import QueryRequest, retrieve
from repo_manager import clone_single_repo, resolve_repos, sync_and_get_commit, read_last_ingested_commit, write_last_ingested_commit

cache: dict[str, dict] = {}
CACHE_TTL = 3600

STATUS_FILE = Path(__file__).resolve().parent.parent / "cloned_repos" / "ingestion_status.json"


def get_status(repo_name: str) -> str:
    if STATUS_FILE.exists():
        data = json.loads(STATUS_FILE.read_text())
        return data.get(repo_name, "unknown")
    return "unknown"


def set_status(repo_name: str, status: str) -> None:
    data = {}
    if STATUS_FILE.exists():
        data = json.loads(STATUS_FILE.read_text())
    data[repo_name] = status
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps(data))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure all loggers propagate to stderr at INFO level
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        root.addHandler(logging.StreamHandler())
    log.info("Starting up — pre-loading embedder...")
    get_embedder()  # loads model into RAM during startup
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
    repo_url: Optional[str] = None
    github_token: Optional[str] = None
    dry_run: bool = False


class QueryResult(BaseModel):
    answer: str


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
      const answerEl = document.createElement('div');
      answerEl.className = 'answer';
      const botMsg = document.createElement('div');
      botMsg.className = 'msg bot';
      botMsg.appendChild(answerEl);
      chat.appendChild(botMsg);
      let pending = '';
      let scheduled = false;
      function flush() {
        if (pending) {
          answerEl.innerHTML += mdToHTML(escapeHTML(pending));
          pending = '';
          chat.scrollTop = chat.scrollHeight;
        }
        scheduled = false;
      }
      try {
        const r = await fetch('/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: q })
        });
        const reader = r.body.getReader();
        const decoder = new TextDecoder();
        let buf = '';
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
          const lines = buf.split('\n');
          buf = lines.pop() || '';
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6);
              if (data === '[DONE]') continue;
              pending += data;
              if (!scheduled) { scheduled = true; requestAnimationFrame(flush); }
            }
          }
        }
        flush();
        typing.style.display = 'none';
      } catch(e) {
        typing.style.display = 'none';
        addMsg('Error: ' + e.message, false);
      }
    }

    input.addEventListener('keydown', e => { if (e.key === 'Enter') ask(); });
  </script>
</body>
</html>"""


@app.post("/index")
def index_repo(request: IngestRequest):
    if not request.repo_url:
        raise HTTPException(status_code=400, detail="repo_url required")
    repo_name, repo_path = clone_single_repo(request.repo_url, request.github_token)
    return {"status": "cloned", "repo": repo_name}


@app.get("/warm")
def warm():
    get_embedder()
    return {"status": "warm"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/reset")
def reset_collection():
    client = get_qdrant_client()
    client.reset_collection()
    return {"status": "reset", "collection": client.settings.collection_name}


@app.post("/ingest")
def ingest(request: IngestRequest, background_tasks: BackgroundTasks):
    if request.repo_url:
        repo_name = request.repo_url.rstrip("/").removesuffix(".git").rsplit("/", 1)[-1]
        set_status(repo_name, "indexing")
        background_tasks.add_task(_run_ingestion, request, repo_name)
        return {"status": "indexing", "repo": repo_name}

    total = ingest_all(target_repo=request.repo, dry_run=request.dry_run)
    return {"status": "ok", "chunks": total}


def _run_ingestion(request: IngestRequest, repo_name: str):
    try:
        name, path = clone_single_repo(request.repo_url, request.github_token)
        current_commit = sync_and_get_commit(path)

        if current_commit == read_last_ingested_commit(name):
            set_status(repo_name, "ready")
            log.info("Skipping ingest for %s — no new commits", name)
            return

        total = ingest_repo(name, path, dry_run=request.dry_run)
        write_last_ingested_commit(name, current_commit)
        set_status(repo_name, "ready")
        log.info("Background ingestion complete — %d chunks indexed for %s", total, name)
    except Exception as e:
        set_status(repo_name, f"error: {e}")
        log.error("Background ingestion failed for %s: %s", repo_name, e)


def _run_ingestion_url(repo_url: str, github_token: str | None, repo_name: str):
    """Shared helper: clone + ingest from URL (used by webhook and /ingest)."""
    try:
        name, path = clone_single_repo(repo_url, github_token)
        total = ingest_repo(name, path)
        set_status(repo_name, "ready")
        log.info("Webhook ingestion complete for %s — %d chunks indexed", repo_name, total)
    except Exception as e:
        set_status(repo_name, f"error: {e}")
        log.error("Webhook ingestion failed for %s: %s", repo_name, e)


@app.post("/webhook/github")
async def github_webhook(payload: dict, background_tasks: BackgroundTasks):
    repo_url = payload.get("repository", {}).get("clone_url")
    repo_name = payload.get("repository", {}).get("name")
    if not repo_url or not repo_name:
        return {"ok": False}
    set_status(repo_name, "indexing")
    background_tasks.add_task(_run_ingestion_url, repo_url, None, repo_name)
    return {"ok": True}


@app.get("/ingest/status/{repo_name}")
def get_ingest_status(repo_name: str):
    return {"status": get_status(repo_name)}


GREETINGS = {"hello", "hi", "hey", "good morning", "good afternoon", "good evening", "thanks", "thank you"}

@app.post("/query")
def query(request: QueryRequest):
    q = request.question.lower().strip()
    if q in GREETINGS or len(q) < 3:
        return StreamingResponse(
            iter([f"data: Hello! I'm your code assistant. Ask me anything about the codebase — what a component does, how a feature works, or where something is defined.\n\ndata: [DONE]\n\n"]),
            media_type="text/event-stream",
        )

    session_id = request.session_id or "default"
    cache_key = f"{session_id}::{hashlib.md5(q.encode()).hexdigest()}"
    if cache_key in cache and time.time() - cache[cache_key]["ts"] < CACHE_TTL:
        cached_answer = cache[cache_key]["answer"]
        return StreamingResponse(
            iter([f"data: {cached_answer}\n\ndata: [DONE]\n\n"]),
            media_type="text/event-stream",
        )

    try:
        t0 = time.time()
        chunks = retrieve(request.question, target_repo=request.target_repo, top_k=request.top_k)
        print(f"retrieve: {time.time()-t0:.2f}s")

        history = session_history.setdefault(session_id, [])

        def generate():
            full_answer = ""
            for token in stream_answer(request.question, chunks, history=history):
                if token.startswith("data: "):
                    content = token[6:]
                    if content != "[DONE]\n\n":
                        full_answer += content
                yield token

            history.append({"role": "user", "content": request.question})
            history.append({"role": "assistant", "content": full_answer.strip()})
            cache[cache_key] = {"answer": full_answer.strip(), "ts": time.time()}

        return StreamingResponse(generate(), media_type="text/event-stream")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/session/{session_id}/clear")
def clear_session(session_id: str) -> dict[str, str]:
    session_history.pop(session_id, None)
    return {"status": "ok", "session_id": session_id}
