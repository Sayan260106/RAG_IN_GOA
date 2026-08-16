from typing import List, Dict, Any
from .loader import DocumentLoader
from .cleaner import TextCleaner
from .deduplicator import Deduplicator

class IngestionProcessor:
    def __init__(self):
        self.loader = DocumentLoader()
        self.cleaner = TextCleaner()
        self.deduplicator = Deduplicator()

    def process(self, dir_path: str) -> List[Dict[str, Any]]:
        docs = self.loader.load_directory(dir_path)
        for doc in docs:
            doc["content"] = self.cleaner.clean(doc["content"])
        return self.deduplicator.deduplicate(docs)
