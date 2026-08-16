"""
Multilingual Cross-Encoder Reranker
-----------------------------------
Uses:
    BAAI/bge-reranker-v2-m3

Designed for:
    FAISS + BM25 + RRF candidate retrieval

Features:
    - Lazy model loading
    - Multilingual reranking
    - Safe fallback if reranker fails
    - Configurable top_k
    - CPU-friendly inference
    - Preserves retrieval metadata
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class MultilingualReranker:
    """
    Multilingual Cross-Encoder reranker.

    The reranker receives candidates from hybrid retrieval
    and assigns a relevance score to each candidate.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        enabled: bool = True,
        top_n: int = 5,
        max_length: int = 512,
        device: str = "cpu",
        batch_size: int = 4,
        fallback_on_error: bool = True,
    ):
        self.model_name = model_name
        self.enabled = enabled
        self.top_n = max(1, int(top_n))
        self.max_length = max_length
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self.fallback_on_error = fallback_on_error

        self._model = None
        self._load_failed = False

    # =========================================================
    # MODEL LOADING
    # =========================================================

    def _load_model(self) -> None:
        """
        Lazily load the CrossEncoder.

        The model is loaded only when reranking is required.
        """

        if not self.enabled:
            return

        if self._model is not None:
            return

        if self._load_failed:
            return

        try:
            from sentence_transformers import CrossEncoder

            logger.info(
                "Loading reranker model '%s' on %s",
                self.model_name,
                self.device,
            )

            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self.device,
            )

            logger.info(
                "Reranker loaded successfully."
            )

        except Exception as exc:
            self._load_failed = True

            logger.exception(
                "Could not load reranker '%s'.",
                self.model_name,
            )

            if not self.fallback_on_error:
                raise RuntimeError(
                    f"Could not load reranker model "
                    f"'{self.model_name}'."
                ) from exc

            logger.warning(
                "Reranker disabled for this runtime. "
                "Falling back to RRF ordering."
            )

    # =========================================================
    # RERANK
    # =========================================================

    def rerank(
        self,
        query: str,
        candidate_docs: List[Dict[str, Any]],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidate documents.

        Parameters
        ----------
        query:
            User query.

        candidate_docs:
            Documents returned by hybrid retrieval/RRF.

        top_n:
            Number of documents to return.
            If omitted, self.top_n is used.

        Returns
        -------
        List[Dict[str, Any]]
            Reranked documents.
        """

        if not candidate_docs:
            return []

        limit = (
            max(1, int(top_n))
            if top_n is not None
            else self.top_n
        )

        limit = min(limit, len(candidate_docs))

        # -----------------------------------------------------
        # Reranking disabled
        # -----------------------------------------------------

        if not self.enabled:
            return self._fallback_results(
                candidate_docs,
                limit,
            )

        # -----------------------------------------------------
        # Empty query
        # -----------------------------------------------------

        if not query or not query.strip():
            return self._fallback_results(
                candidate_docs,
                limit,
            )

        # -----------------------------------------------------
        # Load model
        # -----------------------------------------------------

        self._load_model()

        # Model failed to load
        if self._model is None:
            return self._fallback_results(
                candidate_docs,
                limit,
            )

        # -----------------------------------------------------
        # Build query-document pairs
        # -----------------------------------------------------

        pairs = []

        for document in candidate_docs:

            content = document.get(
                "content",
                document.get(
                    "passage",
                    "",
                ),
            )

            content = str(content).strip()

            pairs.append(
                [
                    query.strip(),
                    content,
                ]
            )

        # -----------------------------------------------------
        # Inference
        # -----------------------------------------------------

        try:
            scores = self._model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )

        except Exception as exc:

            logger.exception(
                "Reranker inference failed."
            )

            if not self.fallback_on_error:
                raise RuntimeError(
                    "Multilingual reranker inference failed."
                ) from exc

            logger.warning(
                "Falling back to RRF ordering."
            )

            return self._fallback_results(
                candidate_docs,
                limit,
            )

        # -----------------------------------------------------
        # Attach scores
        # -----------------------------------------------------

        ranked_documents = []

        for document, score in zip(
            candidate_docs,
            scores,
        ):

            result = document.copy()

            result["rerank_score"] = float(score)

            ranked_documents.append(result)

        # -----------------------------------------------------
        # Sort by reranker score
        # -----------------------------------------------------

        ranked_documents.sort(
            key=lambda doc: doc.get(
                "rerank_score",
                float("-inf"),
            ),
            reverse=True,
        )

        return ranked_documents[:limit]

    # =========================================================
    # FALLBACK
    # =========================================================

    def _fallback_results(
        self,
        candidate_docs: List[Dict[str, Any]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """
        Return candidates in their existing RRF/retrieval order.

        This keeps the RAG pipeline functional even if the
        Cross-Encoder is unavailable.
        """

        results = []

        for document in candidate_docs[:limit]:

            result = document.copy()

            # Keep existing rerank score if one exists.
            if "rerank_score" not in result:

                result["rerank_score"] = float(
                    result.get(
                        "rrf_score",
                        0.0,
                    )
                )

            results.append(result)

        return results

    # =========================================================
    # STATUS
    # =========================================================

    def is_loaded(self) -> bool:
        """Return whether the reranker model is loaded."""

        return self._model is not None

    def is_available(self) -> bool:
        """
        Return whether reranking is currently available.
        """

        return (
            self.enabled
            and self._model is not None
            and not self._load_failed
        )

    # =========================================================
    # UNLOAD
    # =========================================================

    def unload(self) -> None:
        """
        Release the reranker model from memory.
        """

        self._model = None
        self._load_failed = False

        logger.info(
            "Multilingual reranker unloaded."
        )