# code-explainer

Ask plain-English questions about any codebase. Get answers grounded in your actual code, with sources so you know where the info came from.

## Example

**You ask:** *"How does the trust score work?"*

**The bot answers:**

> **Summary**
>
> The trust score is displayed as a percentage using a visual dial that animates from zero to the final score. The dial changes color based on the score value to indicate different levels of trust, ranging from critical to safe.
>
> **Business Impact**
>
> Users can quickly identify the authenticity of media through a color-coded system where higher scores indicate safer content and lower scores signal higher risk.
>
> **Sources**
> 1. components/trustcheck/trust-score-dial.tsx — Confidence: High
> 2. app/page.tsx — Confidence: High

---

## Deploy on Railway (1-click)

Push this repo to GitHub, then:

1. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**
2. Railway detects `docker-compose.yml` and creates two services
3. Remove the auto-created `qdrant` service, click **New** → **Database** → **Add Qdrant** (managed database)
4. In the `rag-backend` service → **Variables**, add:
   - `QDRANT_HOST` — copy from the Qdrant plugin's `QDRANT_URL` (remove `http://`)
   - `QDRANT_PORT` — `6333`
   - `OPENROUTER_API_KEY`
   - `GEMINI_API_KEY`
   - `MONOREPO_URL` — `https://github.com/your-org/your-repo`
   - `HF_TOKEN`
   - `GITHUB_TOKEN`
   - `LLM_MODEL` — `google/gemma-4-31b-it:free`
5. Railway builds and deploys. Once done, you get a public URL.
6. Open the URL in your browser — you'll see the API docs at `/docs`.
7. To index your code, run once: `curl -X POST https://your-url.up.railway.app/ingest -H "Content-Type: application/json" -d '{}'`

---

## Run locally

### Option 1 — Docker (easiest)

```bash
# 1. Fill in your API keys
cp .env.example .env

# 2. Start
docker compose up --build -d

# 3. Index your code
curl -X POST http://localhost:8000/ingest -H "Content-Type: application/json" -d '{}'

# 4. Ask a question
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"question":"What does this project do?"}'

# 5. Stop
docker compose down
```

### Option 2 — Manual (Python)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r rag_backend/requirements.txt
uvicorn rag_backend.main:app --reload
```
