from typing import List, Dict, Any
from .base import BaseChunker

class SemanticChunker(BaseChunker):
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold

    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        paragraphs = text.split("\n\n")
        chunks = []
        for p in paragraphs:
            if p.strip():
                chunks.append({
                    "content": p.strip(),
                    "type": "semantic",
                    "metadata": metadata or {}
                })
        return chunks
