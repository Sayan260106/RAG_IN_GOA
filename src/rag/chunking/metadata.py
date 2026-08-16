from typing import Dict, Any

class MetadataEnricher:
    def enrich(self, chunk: Dict[str, Any], doc_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        enriched = dict(chunk)
        meta = enriched.get("metadata", {})
        meta.update(doc_metadata or {})
        meta["length"] = len(chunk.get("content", ""))
        enriched["metadata"] = meta
        return enriched
