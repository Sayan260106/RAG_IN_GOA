"""
BM25 Lexical Retrieval Store
----------------------------
Sparse lexical retrieval using BM25Okapi.

Used together with:
    FAISS dense retrieval
    Reciprocal Rank Fusion (RRF)
    Cross-Encoder reranking

Designed to work with English and Indic text.
"""

import logging
import re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def tokenize(text: str) -> List[str]:
    """
    Lightweight multilingual tokenizer.

    Keeps Unicode word characters so that Indic scripts are not
    discarded.
    """

    if not text:
        return []

    return re.findall(
        r"\b\w+\b",
        text.lower(),
        flags=re.UNICODE,
    )


class BM25Store:
    """
    In-memory BM25 lexical retrieval store.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.k1 = k1
        self.b = b

        self.corpus: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []

        self.bm25 = None

    def add_documents(
        self,
        docs: List[Dict[str, Any]],
    ) -> None:
        """
        Add documents to the BM25 corpus.

        Each document should contain:
            id
            content
        """

        if not docs:
            return

        for doc in docs:
            content = doc.get(
                "content",
                doc.get("passage", ""),
            )

            if not content:
                logger.warning(
                    "Skipping document without content: %s",
                    doc.get("id", "unknown"),
                )
                continue

            self.corpus.append(
                doc.copy()
            )

            self.tokenized_corpus.append(
                tokenize(content)
            )

        self._rebuild_index()

        logger.info(
            "BM25 index updated. Total documents: %d",
            len(self.corpus),
        )

    def _rebuild_index(self) -> None:
        """
        Build/rebuild BM25Okapi index.
        """

        if not self.tokenized_corpus:
            self.bm25 = None
            return

        try:
            from rank_bm25 import BM25Okapi

            self.bm25 = BM25Okapi(
                self.tokenized_corpus,
                k1=self.k1,
                b=self.b,
            )

        except ImportError as exc:
            raise RuntimeError(
                "rank-bm25 is not installed. "
                "Install it using: pip install rank-bm25"
            ) from exc

        except Exception:
            logger.exception(
                "Failed to build BM25 index."
            )
            raise

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search the BM25 index.

        Returns documents containing:
            bm25_score
            normalized_bm25
        """

        if not self.corpus:
            return []

        if top_k <= 0:
            return []

        top_k = min(
            int(top_k),
            len(self.corpus),
        )

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        if self.bm25 is None:
            self._rebuild_index()

        scores = self.bm25.get_scores(
            query_tokens
        )

        # Get highest-scoring documents.
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        # Normalize scores for easier inspection.
        max_score = max(scores) if len(scores) else 0.0

        results: List[Dict[str, Any]] = []

        for index in ranked_indices:
            document = self.corpus[index].copy()

            score = float(scores[index])

            document["bm25_score"] = score

            if max_score > 0:
                document["normalized_bm25"] = (
                    score / max_score
                )
            else:
                document["normalized_bm25"] = 0.0

            results.append(document)

        return results

    def count(self) -> int:
        """Return number of indexed documents."""

        return len(self.corpus)

    def clear(self) -> None:
        """Clear the BM25 index and documents."""

        self.corpus.clear()
        self.tokenized_corpus.clear()
        self.bm25 = None

        logger.info(
            "BM25 index cleared."
        )