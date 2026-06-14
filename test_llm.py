from rag_backend.retriever import retrieve
from rag_backend.llm import generate_answer

chunks = retrieve('How does authentication work from a user perspective?', top_k=5)
print(f'Retrieved {len(chunks)} chunks')

answer = generate_answer('How does authentication work from a user perspective?', chunks)
print('=== ANSWER ===')
print(answer)
print()
print('=== SOURCES ===')
for c in chunks:
    print(f"  {c.repo_name}/{c.file_path} (lines {c.start_line}-{c.end_line})")
