import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "rag_backend"))

from rag_backend.retriever import retrieve, RetrievedChunk, _rerank, _rrf_merge, _deduplicate_exact


def test_retrieval_returns_results():
    chunks = retrieve("How does authentication work?", top_k=5)
    assert len(chunks) > 0, "Should return at least one chunk"
    for c in chunks:
        assert c.confidence >= 0, f"Confidence should be non-negative, got {c.confidence}"
        assert c.file_path, "Should have a file path"
        assert c.text, "Should have text content"


def test_rerank_produces_differentiated_scores():
    chunks = retrieve("How does authentication work?", top_k=10)
    assert len(chunks) >= 2, "Need at least 2 chunks for rerank test"
    scores = [c.score for c in chunks]
    assert len(set(scores)) > 1, (
        f"Scores should be differentiated after rerank, got all identical: {scores}"
    )


def test_deduplicate_exact():
    chunks = retrieve("How does authentication work?", top_k=10)
    assert len(chunks) >= 2, "Need at least 2 chunks for dedup test"
    duped = chunks + chunks[:2]
    deduped = _deduplicate_exact(duped)
    assert len(deduped) == len(chunks), (
        f"Dedup should remove exact duplicates, got {len(deduped)} from {len(duped)}"
    )


def test_rrf_merge_differentiates():
    chunks = retrieve("How does authentication work?", top_k=5)
    assert len(chunks) > 0, "Need at least 1 chunk for RRF test"
    single_result = [chunks]
    merged = _rrf_merge(single_result)
    assert len(merged) > 0, "RRF merge should return results"


def test_repo_isolation():
    chunks_all = retrieve("How does authentication work?", top_k=5)
    assert len(chunks_all) > 0, "Should have results without repo filter"
    repos = set(c.repo_name for c in chunks_all)
    if len(repos) > 1:
        first_repo = list(repos)[0]
        chunks_filtered = retrieve("How does authentication work?", target_repo=first_repo, top_k=5)
        for c in chunks_filtered:
            assert c.repo_name == first_repo, (
                f"Filtered results should only contain repo '{first_repo}', got '{c.repo_name}'"
            )
