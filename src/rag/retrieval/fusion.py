"""
Reciprocal Rank Fusion (RRF)
----------------------------
Combines multiple ranked retrieval result lists into a single ranking.

Pipeline:

    FAISS
      +
    BM25
      ↓
     RRF
      ↓
 Unified candidates
      ↓
 Cross-Encoder Reranker
"""

import hashlib
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class ReciprocalRankFusion:
    """
    Weighted Reciprocal Rank Fusion.

    RRF formula:

        score(d) = Σ weight_i / (k + rank_i)

    Rank starts at 1.
    """

    def __init__(
        self,
        k: int = 60,
        weights: Optional[List[float]] = None,
    ):
        if k <= 0:
            raise ValueError(
                "RRF k must be greater than 0."
            )

        self.k = int(k)
        self.weights = weights

    def fuse(
        self,
        ranked_lists: List[List[Dict[str, Any]]],
        top_k: int = 10,
        weights: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fuse multiple ranked retrieval lists.

        Example:

            fuse(
                [dense_results, sparse_results],
                top_k=10,
                weights=[0.65, 0.35],
            )
        """

        if not ranked_lists:
            return []

        if top_k <= 0:
            return []

        # -----------------------------------------------------
        # Determine weights
        # -----------------------------------------------------

        active_weights = (
            weights
            if weights is not None
            else self.weights
        )

        if active_weights is None:
            active_weights = [
                1.0
                for _ in ranked_lists
            ]

        if len(active_weights) != len(ranked_lists):
            raise ValueError(
                "Number of RRF weights must match "
                "number of ranked lists."
            )

        # -----------------------------------------------------
        # Validate weights
        # -----------------------------------------------------

        active_weights = [
            float(weight)
            for weight in active_weights
        ]

        if any(weight < 0 for weight in active_weights):
            raise ValueError(
                "RRF weights cannot be negative."
            )

        if sum(active_weights) == 0:
            raise ValueError(
                "At least one RRF weight must be greater than 0."
            )

        # -----------------------------------------------------
        # Fusion structures
        # -----------------------------------------------------

        scores: Dict[str, float] = {}

        doc_map: Dict[str, Dict[str, Any]] = {}

        # -----------------------------------------------------
        # Process ranked lists
        # -----------------------------------------------------

        for list_index, ranked_list in enumerate(
            ranked_lists
        ):

            if not ranked_list:
                continue

            weight = active_weights[list_index]

            for rank, document in enumerate(
                ranked_list,
                start=1,
            ):

                if not isinstance(document, dict):
                    continue

                doc_id = self._get_document_id(
                    document
                )

                # Keep first complete representation.
                if doc_id not in doc_map:
                    doc_map[doc_id] = document.copy()

                contribution = (
                    weight
                    / (self.k + rank)
                )

                scores[doc_id] = (
                    scores.get(doc_id, 0.0)
                    + contribution
                )

        if not scores:
            return []

        # -----------------------------------------------------
        # Sort by RRF score
        # -----------------------------------------------------

        ranked_documents = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        # -----------------------------------------------------
        # Build final results
        # -----------------------------------------------------

        results: List[Dict[str, Any]] = []

        for doc_id, rrf_score in ranked_documents[
            :top_k
        ]:

            document = doc_map[doc_id].copy()

            document["rrf_score"] = float(
                rrf_score
            )

            results.append(document)

        logger.debug(
            "RRF fused %d ranked lists into %d candidates.",
            len(ranked_lists),
            len(results),
        )

        return results

    # =========================================================
    # DOCUMENT ID
    # =========================================================

    @staticmethod
    def _get_document_id(
        document: Dict[str, Any],
    ) -> str:
        """
        Get a deterministic document identifier.

        Priority:

        1. Explicit 'id'
        2. Content hash
        """

        document_id = document.get("id")

        if document_id is not None:
            return str(document_id)

        content = document.get(
            "content",
            document.get(
                "passage",
                "",
            ),
        )

        content = str(content)

        return hashlib.sha256(
            content.encode(
                "utf-8",
                errors="ignore",
            )
        ).hexdigest()