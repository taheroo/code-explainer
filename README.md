# Final RAG

RAG backend that ingests GitHub repos and answers questions about them using LLMs.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r rag_backend/requirements.txt
```

## Configuration

Create `rag_backend/.env` (copy and fill in your keys):

```env
REPO_MODE=monorepo
MONOREPO_URL=https://github.com/your-org/your-repo
OPENROUTER_API_KEY=your_openrouter_key
LLM_MODEL=google/gemma-4-31b-it:free
QDRANT_COLLECTION=codebase
HF_TOKEN=your_hf_token
SCORE_THRESHOLD=0.15
GITHUB_TOKEN=your_github_token
```

| Variable | Description |
|----------|-------------|
| `MONOREPO_URL` | GitHub repo with service subfolders at root |
| `OPENROUTER_API_KEY` | OpenRouter key for LLM |
| `HF_TOKEN` | HuggingFace token for embeddings |
| `GITHUB_TOKEN` | (Optional) For private repos |

## Run

```powershell
uvicorn rag_backend.main:app --reload
```

The server will:
1. Clone `MONOREPO_URL` into `cloned_repos/`
2. Walk all subfolders (skipping `.git`, `node_modules`, `__pycache__`, `.venv`, `dist`, `build`)
3. Ingest each as a repo
4. Serve the API on `http://localhost:8000`

## API

### `POST /query`

Ask a question about your codebase.

**PowerShell:**
```powershell
Invoke-RestMethod -Uri http://localhost:8000/query `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"question":"How does user login work?"}'
```

**curl:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"How does user login work?"}'
```

### Pipeline

1. **Ingest** — Scans repos, chunks code + markdown, embeds via dense + sparse vectors, stores in Qdrant.
2. **Retrieve** — Vector search + cross-encoder reranking with confidence normalization (top result = 100%).
3. **Generate** — LLM (OpenRouter) summarizes results into business-friendly markdown.

### Confidence scoring

Scores normalized relative to the top result:
- **High** > 80% — best match, highly relevant
- **Medium** 50-80% — moderately relevant
- **Low** < 50% — tangentially related

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Ask a question, get answer + sources |
| `/ingest` | POST | Re-index repos |
| `/health` | GET | Health check |
