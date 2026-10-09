from app.services.rag import _extract_used_citations, _filter_by_relevance
from app.providers.vector_store import RetrievedChunk


def test_extract_citations():
    text = "Según el paper [1] y además [3], pero no [99]."
    assert _extract_used_citations(text, 3) == [1, 3]


def test_filter_reindexes():
    chunks = [
        RetrievedChunk("a", "d", "f.pdf", 1, "t1", 0.9, 1),
        RetrievedChunk("b", "d", "f.pdf", 2, "t2", 0.1, 2),
        RetrievedChunk("c", "d", "f.pdf", 3, "t3", 0.5, 3),
    ]
    kept = _filter_by_relevance(chunks, 0.25)
    assert [c.chunk_id for c in kept] == ["a", "c"]
    assert [c.citation_index for c in kept] == [1, 2]
