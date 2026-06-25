# code-explainer

Ask plain-English questions about any GitHub monorepo. Get business-friendly answers grounded in your actual code.

## Quick start

```bash
git clone https://github.com/taheroo/code-explainer
cd code-explainer
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r rag_backend/requirements.txt
```

### macOS / Linux (Python 3)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r rag_backend/requirements.txt
```

Create `rag_backend/.env`:

```env
REPO_MODE=monorepo
MONOREPO_URL=https://github.com/your-org/your-repo
OPENROUTER_API_KEY=sk-or-...       # openrouter.ai
LLM_MODEL=google/gemma-4-31b-it:free
QDRANT_COLLECTION=codebase
HF_TOKEN=hf_...                    # huggingface.co
GITHUB_TOKEN=                      # optional, for private repos
```

```bash
uvicorn rag_backend.main:app --reload
```

Auto-clones your repo, indexes all service folders, serves on `http://localhost:8000`.

### Using Docker

Prerequisites: [Docker](https://docker.com) (with Compose plugin).

```bash
# 1. Create .env from template and fill in your API keys
cp .env.example .env

# 2. Build and start both services (Qdrant + rag-backend)
docker compose up --build -d

# 3. Watch startup logs (models load, then server starts)
docker compose logs -f rag-backend

# 4. Check health
curl http://localhost:8000/health

# 5. Ingest the code into Qdrant
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{}'

# 6. Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What does this project do?"}'

# 7. Stop everything
docker compose down
```

> **Windows users:** PowerShell replaces `curl` with its own alias. Use `curl.exe` or prefix with `cmd /c "..."`. Example: `cmd /c "curl -s http://localhost:8000/health"`.

The `.env` file lives at the project root (not inside `rag_backend/`), and the `cloned_repos/` directory is bind-mounted into the container so manual clones are visible at runtime.

## If you want to clone manually

Clone your repo so its root lands directly in `cloned_repos/` at the project root:

```bash
cd code-explainer
git clone https://github.com/your-org/your-repo cloned_repos
```

If `cloned_repos/` already exists, the auto-cloner skips cloning, so a manual clone works fine.

Do not clone into a nested folder like `cloned_repos/my-repo/` — the engine expects service folders as direct children of `cloned_repos/`, not one level deeper.

## Example response

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"How does the trust score work?"}'
```

```json
{
  "answer": "## Summary\n\nThe trust score is displayed as a percentage using a visual dial that animates from zero to the final score. The dial changes color based on the score value to indicate different levels of trust, ranging from critical to safe.\n\n## Business Impact\n\nUsers can quickly identify the authenticity of media through a color-coded system where higher scores indicate safer content and lower scores signal higher risk.\n\n## Sources\n\n1. components/trustcheck/trust-score-dial.tsx — Confidence: High\n\n   * Explains the animation logic and the color thresholds for safe, accent, warning, and critical scores.\n2. app/page.tsx — Confidence: High\n\n   * Describes the risk profiles associated with the scores, including Critical, High, Moderate, Low, and Verified.",
  "sources": [
    {
      "folder_name": "components",
      "file_path": "components/trustcheck/trust-score-dial.tsx",
      "symbol_name": "TrustScoreDial",
      "language": "tsx",
      "start_line": 10,
      "end_line": 100,
      "confidence": 0.0,
      "confidence_label": "Low"
    },
    {
      "folder_name": "app",
      "file_path": "app/page.tsx",
      "symbol_name": "Home",
      "language": "tsx",
      "start_line": 1,
      "end_line": 50,
      "confidence": 0.0,
      "confidence_label": "Low"
    }
  ]
}
```

## Deployment

The app is deployed on **Railway** with **Qdrant Cloud** for vector storage.

### Live instance

- **URL:** https://rendoohelp-production.up.railway.app
- **Region:** San Francisco (sfo)
- **Qdrant:** Managed cloud instance for persistent vector data

### Architecture

| Component | Service |
|---|---|
| **App server** | Railway — FastAPI (uvicorn) inside Docker |
| **Vector DB** | Qdrant Cloud — dense + sparse hybrid search |
| **LLM** | OpenRouter / Groq (Gemini, LLaMA, Gemma) |
| **Embeddings** | Sentence Transformers (BGE-small) loaded at runtime |

### Environment variables (Railway dashboard)

```env
QDRANT_URL=https://your-qdrant-cloud-instance.cloud.qdrant.io
QDRANT_API_KEY=qdrant-...
OPENROUTER_API_KEY=sk-or-...
GROQ_API_KEY=gsk-...
REPO_MODE=monorepo
MONOREPO_URL=https://github.com/your-org/your-repo
```

The `Dockerfile` builds the backend, the `docker-compose.yml` is used for local development (with a local Qdrant container). On Railway, Qdrant Cloud replaces the local container.

## Pipeline

| Step | What happens |
|---|---|
| **Ingest** | Scans repos, chunks code by function/class (AST for Python, regex for JS/TS), embeds dense + sparse vectors, stores in Qdrant |
| **Retrieve** | Hybrid dense/sparse search → cross-encoder reranking → top 5 chunks |
| **Generate** | LLM (Groq/OpenRouter) summarizes chunks with strict grounding prompt |

## Confidence

Cross-encoder scores are normalized through min-max scaling to [0, 1].

| Label | Confidence |
|---|---|
| High | > 70% |
| Medium | 40–70% |
| Low | < 40% |

## Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/query` | POST | Ask a question, returns answer |
| `/ingest` | POST | Re-index repos (optional `repo` param) |
| `/health` | GET | Server health check |
| `/session/{id}/clear` | GET | Clear chat history |
