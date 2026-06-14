import httpx, json

r = httpx.post('http://localhost:8000/query', json={
    'question': 'What happens when a user creates a reservation?',
    'top_k': 5
}, timeout=120)
print('Status:', r.status_code)
d = r.json()
print('=== ANSWER ===')
print(d.get('answer', 'NO ANSWER'))
print()
print('=== SOURCES ===')
for s in d.get('sources', []):
    print(json.dumps(s))
