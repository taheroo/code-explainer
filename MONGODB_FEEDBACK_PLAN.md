# MongoDB + Feedback Plan

## Goal
Save chat history + user ratings to improve answers over time.

## 1. MongoDB Setup
- Create free cluster at https://mongodb.com/cloud/atlas
- Get connection string
- Add to Railway: `railway variable set --service rendoo_help MONGO_URL="mongodb+srv://..."`

## 2. Code Changes

### New file `rag_backend/db.py`
```python
from pymongo import MongoClient
import os

client = MongoClient(os.getenv("MONGO_URL"))
chat_col = client.rag.chat_history
feedback_col = client.rag.feedback
```

### New endpoints in `main.py`
- `POST /feedback` — body: `{question, answer, rating(1-5), comment?}`
- `GET /history/{session_id}` — returns chat history
- `GET /feedback/export` — download all feedback as JSONL

### Auto-save in `/query`
```python
chat_col.insert_one({
    "session_id": session_id,
    "question": question,
    "answer": answer,
    "sources": sources,
    "timestamp": datetime.utcnow()
})
```

## 3. Dependency
Add to `requirements.txt`:
```
pymongo[srv]
```
