from __future__ import annotations

import os
import time
from typing import Iterable

import httpx
from dotenv import load_dotenv
from pathlib import Path

from .retriever import RetrievedChunk

load_dotenv(Path(__file__).resolve().parent / ".env")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models"
MAX_RETRIES = 3
RETRY_DELAY = 2

SYSTEM_PROMPT = (
    "You are a business assistant explaining software to non-technical teams. "
    "Answer ONLY from provided context. No code, no jargon, no technical terms. "
    "No emojis, no special Unicode symbols (stars, arrows, etc). "
    "If context is insufficient say so clearly.\n\n"
    "Format every response exactly like this:\n\n"
    "## Summary\n\n"
    "[2-3 plain English sentences explaining what happens]\n\n"
    "## Business Impact\n\n"
    "[1-2 sentences on what this means for users or the business]\n\n"
    "## Sources\n\n"
    "1. [file_path] — Confidence: [confidence_label]\n\n"
    "   * [one line saying what this file contributes]\n"
    "2. [file_path] — Confidence: [confidence_label]\n\n"
    "   * [one line saying what this file contributes]"
)


def format_context(chunks: Iterable[RetrievedChunk]) -> str:
    parts: list[str] = []
    for chunk in chunks:
        location = chunk.file_path
        if chunk.symbol_name:
            location = f"{location}::{chunk.symbol_name}"
        conf_pct = round(chunk.confidence * 100, 1)
        label = "High" if conf_pct > 80 else "Medium" if conf_pct >= 50 else "Low"
        parts.append(
            f"--- SOURCE: {location} "
            f"(lines {chunk.start_line}-{chunk.end_line}) "
            f"[confidence={conf_pct}% {label}] ---\n"
            f"{chunk.text}"
        )
    return "\n\n".join(parts)


def _summarize_fallback(chunks: list[RetrievedChunk]) -> str:
    numbered = "\n".join(
        f"{i}. {c.file_path} — Confidence: {round(c.confidence * 100, 1)}%\n\n   * {c.symbol_name or 'block'}"
        for i, c in enumerate(chunks[:5], 1)
    )
    return (
        "## Summary\n\n"
        "The LLM service is temporarily unavailable. Based on the retrieved "
        "code context, relevant information has been found in the source files below.\n\n"
        "## Business Impact\n\n"
        "Unable to generate a natural-language explanation at this time. "
        "Please review the source files directly.\n\n"
        "## Sources\n\n"
        f"{numbered}"
    )


def _call_openrouter(api_key: str, model: str, prompt: str) -> str | None:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    for attempt in range(MAX_RETRIES):
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(OPENROUTER_URL, json=payload, headers=headers)
                if response.status_code == 429 and attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (2 ** attempt))
                    continue
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
        except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError):
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt))
                continue
    return None


def _call_gemini(api_key: str, model: str, prompt: str) -> str | None:
    url = f"{GEMINI_URL}/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": f"{SYSTEM_PROMPT}\n\n{prompt}"}]}],
        "generationConfig": {"temperature": 0.0},
    }
    for attempt in range(MAX_RETRIES):
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload)
                if response.status_code == 429 and attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (2 ** attempt))
                    continue
                response.raise_for_status()
                return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError):
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt))
                continue
    return None


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return (
            "## Summary\n\n"
            "I could not find any relevant information in the codebase "
            "to answer your question. Please try rephrasing it "
            "or ask about a different topic."
        )

    context = format_context(chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {question}"

    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        model = os.getenv("LLM_MODEL", "gemini-2.0-flash")
        result = _call_gemini(gemini_key, model, prompt)
        if result:
            return result
        return _summarize_fallback(chunks)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")
    model = os.getenv("LLM_MODEL", "google/gemma-4-31b-it:free")
    result = _call_openrouter(api_key, model, prompt)
    if result:
        return result
    return _summarize_fallback(chunks)
