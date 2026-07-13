from __future__ import annotations

import concurrent.futures
import json as _json
import logging
import os
import time
from functools import lru_cache
from typing import Generator, Iterable

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
SYSTEM_PROMPT = (
    "You explain code retrieved from a codebase. Answer ONLY from the provided context. "
    "If unsure or the context lacks the information, say so — never invent details.\n\n"
    "Response Guidelines:\n"
    "- Detect the user's knowledge level from their question. Words like 'beginner', 'simple', 'explain like I am 5', "
    "'ELI5', 'basic' signal a non-technical user. Technical terms, code snippets, or requests for 'details' signal an advanced user.\n"
    "- When the user signals a beginner level:\n"
    "  1. Start with a real-world analogy or a concrete example before introducing any concept.\n"
    "  2. Explain the purpose before the implementation — what it does before how it does it.\n"
    "  3. Avoid technical terms unless you explain them immediately in plain language.\n"
    "  4. Do not show code unless the user asks for it.\n"
    "  5. Use comparisons to everyday things (spreadsheets, recipes, mail, filing cabinets, etc.).\n"
    "- When the user signals an advanced or technical level:\n"
    "  1. Provide implementation details, function names, and code references from the context.\n"
    "  2. Use technical terms precisely.\n"
    "  3. Show code snippets when relevant.\n"
    "- When the user gives an explicit format instruction (e.g. 'in one sentence', 'as bullet points', 'in 3 steps'), "
    "follow it exactly — do not add extra content beyond what was requested.\n"
    "- Always follow explicit formatting instructions such as word limits, sentence counts, bullet points, or tone.\n"
    "- Only use information supported by the retrieved context and never invent details."
)


PARENT_DISPLAY_CHARS = 800


# ---------------------------------------------------------------------------
# Token counting — uses tiktoken (cl100k_base ≈ GPT-4 / Llama 3 tokenizer)
# Close enough for context sizing; no HuggingFace auth required.
# ---------------------------------------------------------------------------
import tiktoken as _tiktoken

_enc: object | None = None


def _get_encoder():
    global _enc
    if _enc is None:
        _enc = _tiktoken.get_encoding("cl100k_base")
    return _enc


def _count_tokens(text: str) -> int:
    """Return the token count for a string."""
    return len(_get_encoder().encode(text))


# Budget: Groq free-tier has ~6 KB payload limit.
# tiktoken (cl100k_base) underestimates code tokens vs Llama 3 by ~10-15%,
# so we apply a 15% safety margin on the budget to stay safely under limit.
# Effective budget: ~4700 tokens for chunks + system overhead + question.
TOKEN_BUDGET_FULL = 4700
TOKEN_BUDGET_RETRY = 2200

# Questions shorter than this are narrow/specific — skip expansion.
# Longer questions are broad and benefit from multi-variant search.
EXPANSION_SKIP_THRESHOLD = 20  # tokens


def format_context(chunks: Iterable[RetrievedChunk]) -> str:
    parts: list[str] = []
    for chunk in chunks:
        location = chunk.file_path
        if chunk.symbol_name:
            location = f"{location}::{chunk.symbol_name}"
        ctx = f"--- {location} ---\n{chunk.text}"
        parts.append(ctx)
    return "\n\n".join(parts)


def _trim_chunks_by_tokens(
    chunks: list[RetrievedChunk],
    token_budget: int,
    question_tokens: int,
) -> list[RetrievedChunk]:
    """Greedily include chunks until the token budget is exhausted.

    The budget accounts for system prompt overhead (~200 tokens) and the
    question itself so the final payload stays within Groq's limits.
    """
    SYSTEM_OVERHEAD = 200
    available = token_budget - SYSTEM_OVERHEAD - question_tokens
    if available <= 0:
        return chunks[:1]

    trimmed: list[RetrievedChunk] = []
    total_tokens = 0
    for chunk in chunks:
        chunk_tokens = _count_tokens(chunk.text)
        if total_tokens + chunk_tokens > available and trimmed:
            break
        trimmed.append(chunk)
        total_tokens += chunk_tokens
    return trimmed


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


@lru_cache(maxsize=32)
def expand_query(question: str) -> list[str]:
    import concurrent.futures

    q_tokens = _count_tokens(question)
    if q_tokens >= EXPANSION_SKIP_THRESHOLD:
        log.info("Question is %d tokens (>= %d) — skipping expansion, using original only", q_tokens, EXPANSION_SKIP_THRESHOLD)
        return [question]

    system_prompt = "You are a query expansion assistant. Output only the variant queries, one per line."
    prompt = (
        f"Generate 4 alternative search queries for a code search engine "
        f"given the original question below. Each variant should use different "
        f"technical terminology to cover different ways the code might express "
        f"the same concept. Return one query per line, no numbering or prefixes.\n\n"
        f"Original: {question}"
    )

    def _try_expand():
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
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_try_expand)
        try:
            result = future.result(timeout=5)
        except concurrent.futures.TimeoutError:
            log.warning("expand_query timed out after 5s — falling back to [question]")
            result = None

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


MAX_HISTORY_TURNS = 5


def stream_answer(question: str, chunks: list[RetrievedChunk], history: list[dict] | None = None) -> Generator[str, None, None]:
    if not chunks:
        yield "data: I could not find any relevant information in the codebase to answer your question. Please try rephrasing it or ask about a different topic.\n\n"
        yield "data: [DONE]\n\n"
        return

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        yield "data: LLM service not configured. Set GROQ_API_KEY.\n\n"
        yield "data: [DONE]\n\n"
        return

    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    question_tokens = _count_tokens(question)

    # Budgets to try: full token budget first, then a single hard retry.
    for attempt, budget in enumerate([TOKEN_BUDGET_FULL, TOKEN_BUDGET_RETRY]):
        trimmed = _trim_chunks_by_tokens(chunks, budget, question_tokens)
        context = format_context(trimmed)
        prompt = f"Context:\n{context}\n\nQuestion: {question}"

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(history[-MAX_HISTORY_TURNS * 2:])
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {groq_key}",
            "Content-Type": "application/json",
        }
        try:
            with httpx.Client(timeout=httpx.Timeout(120.0, connect=15.0)) as client:
                with client.stream("POST", GROQ_URL, json=payload, headers=headers) as resp:
                    if resp.status_code == 413:
                        if attempt == 0:
                            log.warning("413 Payload Too Large (budget=%d tokens) — retrying with %d tokens", budget, TOKEN_BUDGET_RETRY)
                            continue
                        else:
                            log.error("413 Payload Too Large on retry (budget=%d tokens) — giving up", budget)
                            yield "data: The context is too large for the LLM to process. Try a shorter question or ask about a specific file.\n\n"
                            yield "data: [DONE]\n\n"
                            return
                    for line in resp.iter_lines():
                        if line.startswith("data: "):
                            data = line[6:].strip()
                            if data == "[DONE]":
                                break
                            try:
                                event = _json.loads(data)
                                token = event["choices"][0]["delta"].get("content", "")
                                if token:
                                    yield f"data: {token}\n\n"
                            except _json.JSONDecodeError:
                                pass
        except Exception as e:
            yield f"data: Error: {e}\n\n"
            break

    yield "data: [DONE]\n\n"


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return (
            "I could not find any relevant information in the codebase "
            "to answer your question. Please try rephrasing it "
            "or ask about a different topic."
        )

    question_tokens = _count_tokens(question)
    trimmed = _trim_chunks_by_tokens(chunks, TOKEN_BUDGET_FULL, question_tokens)
    context = format_context(trimmed)
    prompt = f"Context:\n{context}\n\nQuestion: {question}"

    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_call_groq, groq_key, model, prompt)
            try:
                result = future.result(timeout=12)
            except concurrent.futures.TimeoutError:
                log.warning("Groq generate_answer timed out after 12s")
                result = None
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
