# Project Summary — RAG Backend (code-explainer)

## Architecture
- **2 Docker services**: Qdrant (vector DB) + FastAPI app
- Connected via `QDRANT_HOST=qdrant` env var
- Pre-downloaded ML models in Dockerfile for fast startup

## Files

| File | Purpose |
|---|---|
| `rag_backend/main.py` | FastAPI app + embedded chat UI at `GET /` |
| `rag_backend/Dockerfile` | Multi-stage build (CPU torch, model pre-download) |
| `docker-compose.yml` | qdrant + rag-backend services |
| `.env.example` | Template for required env vars |
| `.dockerignore` | Excludes caches, secrets from build |
| `rag_backend/repo_manager.py` | Git clone logic + repo discovery |
| `rag_backend/ingest.py` | Code chunking + Qdrant indexing |
| `rag_backend/retriever.py` | Hybrid search + cross-encoder reranking |
| `rag_backend/llm.py` | LLM answer generation (Gemini/OpenRouter) |
| `rag_backend/requirements.txt` | Python dependencies |

## Dockerfile Key Points
- Stage 1 (builder): install git, CPU torch, Python deps, pre-download models (`BAAI/bge-small-en-v1.5`, `cross-encoder/ms-marco-MiniLM-L6-v2`)
- Stage 2 (final): fresh slim image, copy only site-packages + model cache + code
- `CMD uvicorn rag_backend.main:app --host 0.0.0.0 --port 8000`
- Context = project root, Dockerfile = `rag_backend/Dockerfile`

## docker-compose.yml Key Points
- `qdrant` service: image `qdrant/qdrant:latest`, port 6333, named volume for storage
- `rag-backend` service: builds from Dockerfile, port 8000, env vars from `.env`
- `cloned_repos` uses **bind mount** (`./cloned_repos:/app/cloned_repos`) — host directory visible in container
- `hf_cache` uses named volume for HuggingFace cache

## Env Variables (`.env`)
```
REPO_MODE=monorepo
MONOREPO_URL=https://github.com/your-org/your-repo
OPENROUTER_API_KEY=sk-or-...
GEMINI_API_KEY=...
LLM_MODEL=google/gemma-4-31b-it:free
QDRANT_COLLECTION=codebase
HF_TOKEN=hf_...
GITHUB_TOKEN=ghp_...   # optional
```

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Chat UI (HTML) |
| GET | `/health` | Health check |
| POST | `/ingest` | Index code (body: `{"repo": null, "dry_run": false}`) |
| POST | `/query` | Ask question (body: `{"question": "..."}`) |
| GET | `/session/{id}/clear` | Clear chat history |

## Local Testing
```bash
cp .env.example .env   # fill in keys
docker compose up --build -d
curl http://localhost:8000/health
curl -X POST http://localhost:8000/ingest -H "Content-Type: application/json" -d '{}'
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"question":"What does this project do?"}'
docker compose down
```

## Railway Deployment
1. New Project → Deploy from GitHub repo
2. Remove auto-created `qdrant` → Add Qdrant plugin (managed DB)
3. Set env vars in rag-backend service
4. After deploy, open URL → chat UI at root, Swagger at `/docs`
5. Hit `/ingest` once, then chat

## Key Decisions
- **CPU torch** instead of GPU (models are small, image stays under 2GB)
- **Railway** over Render/Vercel: Railway has native docker-compose + managed Qdrant
- **Bind mount** for cloned_repos (named volume would be empty)
- **Chat UI in FastAPI** — single file, no separate frontend needed, works on Railway
- **Auto-clone on startup** — `resolve_repos()` clones from `MONOREPO_URL` if `cloned_repos/` is empty
