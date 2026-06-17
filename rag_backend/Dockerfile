FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5'); SentenceTransformer('cross-encoder/ms-marco-MiniLM-L6-v2')"

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /root/.cache /root/.cache
COPY --from=builder /app /app

EXPOSE 8000

CMD ["uvicorn", "rag_backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
