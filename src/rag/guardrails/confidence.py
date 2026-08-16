from typing import List, Dict, Any

class ConfidenceScorer:
    def calculate_confidence(self, retrieval_scores: List[float], query: str) -> float:
        if not retrieval_scores:
            return 0.75
        avg_score = sum(retrieval_scores) / len(retrieval_scores)
        return min(round(avg_score * 0.95 + 0.05, 2), 0.99)
