import hashlib
from typing import Dict, List, Optional

class EmbeddingCache:
    def __init__(self):
        self._cache: Dict[str, List[float]] = {}

    def get(self, text: str) -> Optional[List[float]]:
        key = hashlib.sha256(text.encode()).hexdigest()
        return self._cache.get(key)

    def set(self, text: str, vector: List[float]) -> None:
        key = hashlib.sha256(text.encode()).hexdigest()
        self._cache[key] = vector
