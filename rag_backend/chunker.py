from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

MIN_CHUNK_CHARS = 40


def chunk_code(file_path: str, source_code: str) -> list[dict[str, Any]]:
    suffix = Path(file_path).suffix.lower()

    if "readme" in Path(file_path).stem.lower():
        return [{
            "text": source_code.strip(),
            "start": 1,
            "end": len(source_code.splitlines()),
            "symbol": "readme",
        }]

    if suffix == ".py":
        chunks = _chunk_python(source_code)
    elif suffix in {".js", ".jsx", ".ts", ".tsx"}:
        chunks = _chunk_braced_source(source_code)
    else:
        chunks = []

    cleaned = [chunk for chunk in chunks if _is_useful_chunk(chunk)]
    return cleaned or _fallback_line_chunks(source_code)


def _is_useful_chunk(chunk: dict[str, Any]) -> bool:
    text = str(chunk.get("text", "")).strip()
    return len(text) >= MIN_CHUNK_CHARS


def _chunk_python(source_code: str) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []

    lines = source_code.splitlines()
    chunks: list[dict[str, Any]] = []

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start = getattr(node, "lineno", 0)
        end = getattr(node, "end_lineno", start)
        if start <= 0 or end <= 0:
            continue
        text = "\n".join(lines[start - 1:end]).strip()
        if text:
            chunks.append({
                "text": text,
                "start": start,
                "end": end,
                "symbol": getattr(node, "name", node.__class__.__name__),
            })

    return chunks


def _chunk_braced_source(source_code: str) -> list[dict[str, Any]]:
    """Structural regex-based chunker for JS/TS (not AST). Detects function/class/const
    declarations by regex, then tracks brace depth to extract the full body."""
    lines = source_code.splitlines()
    chunks: list[dict[str, Any]] = []
    start_index: int | None = None
    brace_depth = 0
    symbol_name = "block"

    start_pattern = re.compile(
        r"""
        ^\s*
        (?:
            (?:export\s+)?(?:async\s+)?function\s+(?P<fn>[A-Za-z_][A-Za-z0-9_]*) |
            (?:export\s+)?class\s+(?P<class>[A-Za-z_][A-Za-z0-9_]*) |
            (?:export\s+)?const\s+(?P<const>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:async\s*)?(?:function\s*\(|\([^)]*\)\s*=>)
        )
        """,
        re.VERBOSE,
    )

    for idx, line in enumerate(lines, start=1):
        if start_index is None:
            match = start_pattern.search(line)
            if not match:
                continue
            start_index = idx
            symbol_name = next((value for value in match.groupdict().values() if value), "block")

        brace_depth += line.count("{")
        brace_depth -= line.count("}")

        if start_index is not None and brace_depth <= 0 and "{" in "\n".join(lines[start_index - 1:idx]):
            text = "\n".join(lines[start_index - 1:idx]).strip()
            if text:
                chunks.append({
                    "text": text,
                    "start": start_index,
                    "end": idx,
                    "symbol": symbol_name,
                })
            start_index = None
            brace_depth = 0
            symbol_name = "block"

    return chunks


def _fallback_line_chunks(source_code: str) -> list[dict[str, Any]]:
    lines = source_code.splitlines()
    chunks: list[dict[str, Any]] = []
    window_size = 40

    for i in range(0, len(lines), window_size):
        text_block = "\n".join(lines[i:i + window_size]).strip()
        if text_block and len(text_block) >= MIN_CHUNK_CHARS:
            chunks.append({
                "text": text_block,
                "start": i + 1,
                "end": min(i + window_size, len(lines)),
                "symbol": "line_block",
            })

    return chunks

