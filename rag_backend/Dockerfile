# STAGE 1: Build & Dependency Gathering (force rebuild)
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /app

ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

# 1. Torch first (torch cpu not for gpu for size constraints)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2. Requirements second 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Download models third (baked into image — no download at runtime)
RUN python -c "
from sentence_transformers import SentenceTransformer, CrossEncoder
SentenceTransformer('BAAI/bge-small-en-v1.5')
CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
print('Models cached')
"

# 4. Copy your source code
COPY . .

# STAGE 2: Lightweight Final Runtime
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy uvicorn and other CLI binaries from the builder stage
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy python dependencies, huggingface caches, and codebase from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /app /app

EXPOSE 8000

# Fire up the lightweight production server
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}