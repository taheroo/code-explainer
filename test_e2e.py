import httpx, json

base = "http://localhost:8000"

# 1. Root
r = httpx.get(f"{base}/")
print(f"GET / → {r.status_code}")
print(r.json())
print()

# 2. Health
r = httpx.get(f"{base}/health")
print(f"GET /health → {r.status_code}")
print(r.json())
print()

# 3. Query - reservation
r = httpx.post(f"{base}/query", json={
    "question": "How does the reservations service work?",
    "top_k": 5
}, timeout=120)
print(f"POST /query → {r.status_code}")
d = r.json()
print("=== ANSWER ===")
print(d.get("answer", "NO ANSWER"))
print()
print("=== SOURCES ===")
for s in d.get("sources", []):
    print(f"  {s['folder_name']}/{s['file_path']}  (lines {s.get('start_line',0)}-{s.get('end_line',0)})")
print()

# 4. Query - auth
r = httpx.post(f"{base}/query", json={
    "question": "What happens when a user registers?",
    "top_k": 3
}, timeout=120)
print(f"POST /query (auth) → {r.status_code}")
d2 = r.json()
print("=== ANSWER ===")
print(d2.get("answer", "NO ANSWER"))
print()
print("=== SOURCES ===")
for s in d2.get("sources", []):
    print(f"  {s['folder_name']}/{s['file_path']}  (lines {s.get('start_line',0)}-{s.get('end_line',0)})")
