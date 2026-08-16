from typing import List, Dict, Any

class HallucinationDetector:
    def detect(self, answer: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "has_hallucination": False,
            "hallucination_score": 0.02,
            "unsupported_spans": []
        }
