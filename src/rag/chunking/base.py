from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        pass
