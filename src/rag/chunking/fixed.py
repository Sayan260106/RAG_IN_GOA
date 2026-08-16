from typing import List, Dict, Any
from .base import BaseChunker

class FixedSizeChunker(BaseChunker):
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append({
                "content": text[start:end],
                "start": start,
                "end": end,
                "metadata": metadata or {}
            })
            start += self.chunk_size - self.overlap
        return chunks
