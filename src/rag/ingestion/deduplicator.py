import hashlib
from typing import List, Dict, Any

class Deduplicator:
    def __init__(self):
        self.seen_hashes = set()

    def deduplicate(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        unique_docs = []
        for doc in documents:
            content_hash = hashlib.md5(doc.get("content", "").encode()).hexdigest()
            if content_hash not in self.seen_hashes:
                self.seen_hashes.add(content_hash)
                unique_docs.append(doc)
        return unique_docs
