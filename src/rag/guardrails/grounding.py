"""
Guardrails and grounding checks for the HHGoa RAG pipeline.
"""

import re
import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

OFF_TOPIC_RE = re.compile(
    r"(how\s+to\s+make\s+a\s+bomb|hack\s+|exploit|credit\s+card|steal|password\s+crack)",
    re.IGNORECASE,
)


class GuardrailsManager:
    def __init__(
        self,
        min_confidence_threshold: float = 0.70,
        min_context_overlap_ratio: float = 0.20,
    ):
        self.min_confidence_threshold = min_confidence_threshold
        self.min_context_overlap_ratio = min_context_overlap_ratio

    def validate_input(self, query: str) -> Tuple[bool, Optional[str]]:
        if not query or len(query.strip()) < 3:
            return False, "Query is too short or empty."
        if OFF_TOPIC_RE.search(query):
            return False, "Query violates safety policies (unsafe/harmful topic)."
        return True, None

    def evaluate_grounding(
        self,
        query: str,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]],
        raw_similarity: float = 0.0,
    ) -> Dict[str, Any]:
        if not retrieved_chunks:
            return {
                "is_grounded": False,
                "confidence": 0.0,
                "overlap_ratio": 0.0,
                "status": "REJECTED_NO_CONTEXT",
                "reason": "No relevant context chunks found in the knowledge base.",
            }

        context_text = " ".join(
            str(c.get("content", c.get("passage", "")))
            for c in retrieved_chunks
        ).lower()
        context_tokens = set(re.findall(r"\b\w{4,}\b", context_text))
        answer_tokens = set(re.findall(r"\b\w{4,}\b", answer.lower()))

        if not answer_tokens:
            return {
                "is_grounded": False,
                "confidence": 0.0,
                "overlap_ratio": 0.0,
                "status": "REJECTED_EMPTY_ANSWER",
                "reason": "The selected model returned no answer.",
            }

        shared = answer_tokens.intersection(context_tokens)
        overlap_ratio = len(shared) / max(len(answer_tokens), 1)

        confidence = min(
            0.98,
            max(
                0.0,
                (float(raw_similarity) * 0.6)
                + (overlap_ratio * 0.4),
            ),
        )

        grounded = (
            float(raw_similarity) >= self.min_confidence_threshold
            and confidence >= self.min_confidence_threshold
            and overlap_ratio >= self.min_context_overlap_ratio
        )

        return {
            "is_grounded": grounded,
            "confidence": round(confidence, 3),
            "overlap_ratio": round(overlap_ratio, 3),
            "status": "VERIFIED_GROUNDED" if grounded else "FLAGGED_LOW_GROUNDING",
            "reason": (
                None
                if grounded
                else "Generated answer did not meet the grounding requirements."
            ),
        }
