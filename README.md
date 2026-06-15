# code-explainer

Ask plain-English questions about any GitHub monorepo. Get business-friendly answers grounded in your actual code.

## Quick start

```bash
git clone https://github.com/taheroo/code-explainer
cd code-explainer
python -m venv .venv
.venv\Scripts\activate
pip install -r rag_backend/requirements.txt
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

## If you want to clone manually

Clone your repo so its root lands directly in `cloned_repos/` at the project root:

```bash
cd code-explainer
git clone https://github.com/your-org/your-repo cloned_repos
```

If `cloned_repos/` already exists, the auto-cloner skips cloning, so a manual clone works fine.

Do not clone into a nested folder like `cloned_repos/my-repo/` — the engine expects service folders as direct children of `cloned_repos/`, not one level deeper.

## Ask a question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"How does user login work?"}'
```

Response:

```json
{
  "answer": "## Summary\nA user provides email/password…\n\n## Business Impact\nOnly verified users can access…",
  "sources": [{
    "folder_name": "auth-service",
    "file_path": "src/authService.ts",
    "confidence": 0.82,
    "score": 1.52
  }]
}
```

## Pipeline

| Step | What happens |
|---|---|
| **Ingest** | Scans repos, chunks code by function/class (AST for Python, regex for JS/TS), embeds dense + sparse vectors, stores in local Qdrant |
| **Retrieve** | Hybrid dense/sparse search → cross-encoder reranking → top 5 chunks |
| **Generate** | LLM (Gemini or OpenRouter) summarizes chunks with strict grounding prompt |

## Confidence

Cross-encoder scores are normalized through sigmoid: `confidence = 1 / (1 + e^-score)`. All results above `CONFIDENCE_THRESHOLD` (0.0) are returned.

| Label | Confidence |
|---|---|
| High | > 70% |
| Medium | 40–70% |
| Low | < 40% |

## Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/query` | POST | Ask a question, returns answer + sources |
| `/ingest` | POST | Re-index repos (optional `repo` param) |
| `/health` | GET | Server health check |
| `/session/{id}/clear` | GET | Clear chat history |
