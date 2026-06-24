from __future__ import annotations

import logging
import os
import time

import httpx
from dotenv import load_dotenv
from pathlib import Path

log = logging.getLogger(__name__)

from retriever import RetrievedChunk

load_dotenv(Path(__file__).resolve().parent / ".env")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models"

MAX_RETRIES = 2
RETRY_DELAY = 2
SYSTEM_PROMPT = "You are a business translator for non-technical stakeholders (product owners, support, business teams). Answer using ONLY the provided code context. Write a thorough explanation (minimum 3 paragraphs, 150-250 words) in plain business language. Cover: what the feature does from a user perspective, the business problem it solves, and what impact it has on users. No code, no technical jargon, no implementation details. If unsure say so."


PARENT_DISPLAY_CHARS = 800


def format_context(chunks: Iterable[RetrievedChunk]) -> str:
    parts: list[str] = []
    for chunk in chunks:
        location = chunk.file_path
        if chunk.symbol_name:
            location = f"{location}::{chunk.symbol_name}"
        ctx = f"--- {location} ---\n{chunk.text}"
        parts.append(ctx)
    return "\n\n".join(parts)


def _summarize_fallback(chunks: list[RetrievedChunk]) -> str:
    return (
        "The LLM service is temporarily unavailable. Based on the retrieved "
        "code context, relevant information has been found to answer your question. "
        "Please try again later or check the source files directly."
    )


def _call_openrouter(api_key: str, model: str, prompt: str, system_prompt: str | None = None) -> str | None:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
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
                    time.sleep(5 * (2 ** attempt))
                    continue
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
        except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError):
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt))
                continue
    return None


def _call_gemini(api_key: str, model: str, prompt: str, system_prompt: str | None = None) -> str | None:
    url = f"{GEMINI_URL}/{model}:generateContent?key={api_key}"
    sp = system_prompt or SYSTEM_PROMPT
    payload = {
        "contents": [{"parts": [{"text": f"{sp}\n\n{prompt}"}]}],
        "generationConfig": {"temperature": 0.0},
    }
    for attempt in range(MAX_RETRIES):
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload)
                if response.status_code == 429 and attempt < MAX_RETRIES - 1:
                    time.sleep(5 * (2 ** attempt))
                    continue
                response.raise_for_status()
                return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError):
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt))
                continue
    return None


def _call_groq(api_key: str, model: str, prompt: str, system_prompt: str | None = None) -> str | None:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
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
                response = client.post(GROQ_URL, json=payload, headers=headers)
                if response.status_code == 429 and attempt < MAX_RETRIES - 1:
                    time.sleep(5 * (2 ** attempt))
                    continue
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
        except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError) as exc:
            log.warning("Groq API attempt %d/%d failed: %s", attempt + 1, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt))
                continue
    return None


def expand_query(question: str) -> list[str]:
    system_prompt = "You are a query expansion assistant. Output only the variant queries, one per line."
    prompt = (
        f"Generate 4 alternative search queries for a code search engine "
        f"given the original question below. Each variant should use different "
        f"technical terminology to cover different ways the code might express "
        f"the same concept. Return one query per line, no numbering or prefixes.\n\n"
        f"Original: {question}"
    )

    groq_key = os.getenv("GROQ_API_KEY")
    result = None
    if groq_key:
        model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        result = _call_groq(groq_key, model, prompt, system_prompt=system_prompt)
    if not result:
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            model = os.getenv("LLM_MODEL", "gemini-2.0-flash")
            result = _call_gemini(gemini_key, model, prompt, system_prompt=system_prompt)
    if not result:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key:
            model = os.getenv("LLM_MODEL", "google/gemma-4-31b-it:free")
            result = _call_openrouter(api_key, model, prompt, system_prompt=system_prompt)

    if result:
        lines = [line.strip() for line in result.strip().split("\n") if line.strip()]
        seen: set[str] = set()
        variants: list[str] = []
        for q in [question] + lines:
            q_lower = q.lower().strip()
            if q_lower not in seen:
                seen.add(q_lower)
                variants.append(q)
        return variants[:5]
    return [question]


def parse_query(question: str) -> tuple[str, dict[str, str]]:
    system_prompt = "You extract search queries and metadata filters from user questions."
    prompt = (
        f"Extract a clean search query and optional metadata filters from the "
        f"question below. Available filter fields: language (python, js, ts, ...), "
        f"symbol_name (function/class name), file_path (file path pattern).\n\n"
        f"Respond exactly in this format:\n"
        f"QUERY: <cleaned search query>\n"
        f"FILTERS: <key=value, key=value> or NONE if no filters\n\n"
        f"Question: {question}"
    )

    groq_key = os.getenv("GROQ_API_KEY")
    result = None
    if groq_key:
        model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        result = _call_groq(groq_key, model, prompt, system_prompt=system_prompt)
    if not result:
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            model = os.getenv("LLM_MODEL", "gemini-2.0-flash")
            result = _call_gemini(gemini_key, model, prompt, system_prompt=system_prompt)
    if not result:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key:
            model = os.getenv("LLM_MODEL", "google/gemma-4-31b-it:free")
            result = _call_openrouter(api_key, model, prompt, system_prompt=system_prompt)

    if not result:
        return question, {}

    lines = result.strip().split("\n")
    cleaned = question
    filters: dict[str, str] = {}
    for line in lines:
        if line.startswith("QUERY:"):
            cleaned = line[6:].strip()
        elif line.startswith("FILTERS:"):
            filter_str = line[8:].strip()
            if filter_str and filter_str.upper() != "NONE":
                for part in filter_str.split(","):
                    if "=" in part:
                        key, value = part.split("=", 1)
                        filters[key.strip()] = value.strip()
    return cleaned, filters


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return (
            "I could not find any relevant information in the codebase "
            "to answer your question. Please try rephrasing it "
            "or ask about a different topic."
        )

    context = format_context(chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {question}"

    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        result = _call_groq(groq_key, model, prompt)
        if result:
            return result

    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        result = _call_gemini(gemini_key, model, prompt)
        if result:
            return result

    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        model = os.getenv("LLM_MODEL", "google/gemma-4-31b-it:free")
        result = _call_openrouter(api_key, model, prompt)
        if result:
            return result

    return _summarize_fallback(chunks)
