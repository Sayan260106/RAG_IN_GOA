import re
from typing import List, Dict, Any
from .base import BaseChunker

class SentenceChunker(BaseChunker):
    def __init__(self, max_sentences: int = 3):
        self.max_sentences = max_sentences

    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        for i in range(0, len(sentences), self.max_sentences):
            chunk_text = " ".join(sentences[i:i+self.max_sentences])
            chunks.append({
                "content": chunk_text,
                "metadata": metadata or {}
            })
        return chunks
