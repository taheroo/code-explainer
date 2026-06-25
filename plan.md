# Plan — UI Output Cleanup & Extensions (Updated)

## ✅ Task 1 — DONE (June 24)

### What was changed

#### `rag_backend/llm.py`
- `SYSTEM_PROMPT` — removed `## Summary` / `## Sources` formatting instruction
- `_summarize_fallback()` — removed Sources listing + headings, now plain sentence
- No-chunks case — removed `## Summary` heading
- Added `MAX_RETRIES = 1` (was missing, causing crash)

#### `rag_backend/main.py`
- Removed dead `.source` CSS class

### Result
Answers are now plain text with no markdown headings or source file listings.

---

## 👷 Task 2 — Improve Answer Quality (In Progress)

### Problem
Answers are too brief and technical. Need to include **business context / value proposition** — explain the "why" and the business use case, not just the code.

### Changes

#### `rag_backend/llm.py:19` — SYSTEM_PROMPT (v2)
Instruct LLM to give detailed answers including business context:
```
"Answer the question using ONLY the provided code context. Write a detailed, thorough explanation in plain English. Explain what each component does, how they connect, the overall architecture, and the business value. No code snippets. If unsure say so."
```

### Testing
- Test with: `"what does this project do?"`
- Should describe the business idea (trust verification / fraud detection), not just list components

---

## 📋 Upcoming Tasks

- [ ] **MongoDB + feedback** — save chat history + ratings (see `MONGODB_FEEDBACK_PLAN.md`)
- [ ] **Streaming responses** — real-time token streaming in UI
- [ ] **File tree browser** — sidebar showing repo structure
- [ ] **Multi-session support** — switch between conversations
