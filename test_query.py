import httpx, json
r = httpx.post('http://localhost:8000/query', json={'question': 'How does authentication work from a user perspective?', 'top_k': 5}, timeout=120)
print(f'Status: {r.status_code}')
d = r.json()
print('=== ANSWER ===')
print(d.get('answer', 'NO ANSWER'))
print()
print('=== SOURCES ===')
for s in d.get('sources', []):
    print(f"  {s['folder_name']}/{s['file_path']} (lines {s.get('start_line',0)}-{s.get('end_line',0)})")
