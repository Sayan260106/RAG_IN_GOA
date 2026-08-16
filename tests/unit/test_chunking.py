import pytest
from src.rag.chunking.fixed import FixedSizeChunker
from src.rag.chunking.sliding_window import SlidingWindowChunker

def test_fixed_chunker():
    chunker = FixedSizeChunker(chunk_size=100, overlap=10)
    text = "Panaji is the state capital of Goa located on the banks of the Mandovi river estuary."
    chunks = chunker.chunk(text)
    assert len(chunks) >= 1
    assert "Panaji" in chunks[0]["content"]

def test_sliding_window_chunker():
    chunker = SlidingWindowChunker(window_size=10, stride=5)
    text = "Calangute Baga Anjuna Vagator Morjim Ashwem Arambol Mandrem Palolem Agonda"
    chunks = chunker.chunk(text)
    assert len(chunks) >= 1
