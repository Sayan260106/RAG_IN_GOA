import pytest
from src.rag.retrieval.bm25_store import BM25Store
from src.rag.retrieval.fusion import ReciprocalRankFusion

def test_bm25_search():
    store = BM25Store()
    store.index_documents([
        {"id": "1", "content": "Basilica of Bom Jesus Old Goa UNESCO"},
        {"id": "2", "content": "Palolem beach south Goa scenic beach"}
    ])
    results = store.search("Bom Jesus", top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == "1"

def test_rrf_fusion():
    fusion = ReciprocalRankFusion()
    list1 = [{"id": "a"}, {"id": "b"}]
    list2 = [{"id": "b"}, {"id": "c"}]
    fused = fusion.fuse([list1, list2], top_k=2)
    assert len(fused) == 2
    assert fused[0]["id"] == "b"
