from typing import List, Dict, Any
from .base import BaseChunker

class SlidingWindowChunker(BaseChunker):
    def __init__(self, window_size: int = 400, stride: int = 150):
        self.window_size = window_size
        self.stride = stride

    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.stride):
            chunk_words = words[i:i + self.window_size]
            if not chunk_words:
                break
            chunks.append({
                "content": " ".join(chunk_words),
                "token_count": len(chunk_words),
                "metadata": metadata or {}
            })
        return chunks
