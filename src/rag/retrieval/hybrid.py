"""
HHGoa 2026 - Hybrid Retrieval Engine

Pipeline:

    User Query
        |
        +----------------------+
        |                      |
        v                      v
   BGE-M3 / FAISS           BM25
   Dense Retrieval       Sparse Retrieval
        |                      |
        +----------+-----------+
                   |
                   v
        Weighted Reciprocal
           Rank Fusion
                   |
                   v
        BGE Reranker v2-M3
                   |
                   v
           Final Chunks

The retriever is independent from:
    - LLM generation
    - grounding guardrails
    - FastAPI
"""

import hashlib
import logging
import time
from typing import Any, Dict, List, Optional

from src.rag.embeddings.model import (
    MultilingualEmbeddingModel,
)
from src.rag.retrieval.faiss_store import (
    FaissCPUStore,
)
from src.rag.retrieval.bm25_store import (
    BM25Store,
)
from src.rag.retrieval.reranker import (
    MultilingualReranker,
)


logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid FAISS + BM25 retriever using weighted RRF
    followed by multilingual cross-encoder reranking.
    """

    def __init__(
        self,
        embedding_model: Optional[
            MultilingualEmbeddingModel
        ] = None,
        faiss_store: Optional[
            FaissCPUStore
        ] = None,
        bm25_store: Optional[
            BM25Store
        ] = None,
        reranker: Optional[
            MultilingualReranker
        ] = None,
        rrf_k: int = 60,
        dense_weight: float = 0.65,
        sparse_weight: float = 0.35,
        retrieval_multiplier: int = 4,
    ):
        # ======================================================
        # EMBEDDING MODEL
        # ======================================================

        self.embedding_model = (
            embedding_model
            or MultilingualEmbeddingModel(
                model_name="BAAI/bge-m3"
            )
        )

        # ======================================================
        # FAISS
        # ======================================================

        self.faiss_store = (
            faiss_store
            or FaissCPUStore(
                dimension=(
                    self.embedding_model.dimension
                ),
                metric="inner_product",
            )
        )

        # ======================================================
        # BM25
        # ======================================================

        self.bm25_store = (
            bm25_store
            or BM25Store(
                k1=1.5,
                b=0.75,
            )
        )

        # ======================================================
        # RERANKER
        # ======================================================

        self.reranker = (
            reranker
            or MultilingualReranker(
                model_name=(
                    "BAAI/"
                    "bge-reranker-v2-m3"
                ),
                enabled=True,
                top_n=5,
            )
        )

        # ======================================================
        # RRF CONFIG
        # ======================================================

        self.rrf_k = max(
            1,
            int(rrf_k),
        )

        self.dense_weight = float(
            dense_weight
        )

        self.sparse_weight = float(
            sparse_weight
        )

        total_weight = (
            self.dense_weight
            + self.sparse_weight
        )

        if total_weight <= 0:
            self.dense_weight = 0.65
            self.sparse_weight = 0.35

        else:
            self.dense_weight /= total_weight
            self.sparse_weight /= total_weight

        self.retrieval_multiplier = max(
            2,
            int(retrieval_multiplier),
        )

        logger.info(
            "HybridRetriever initialized: "
            "embedding=%s, dimension=%d, "
            "dense_weight=%.3f, "
            "sparse_weight=%.3f, "
            "rrf_k=%d, "
            "retrieval_multiplier=%d",
            getattr(
                self.embedding_model,
                "model_name",
                "unknown",
            ),
            self.embedding_model.dimension,
            self.dense_weight,
            self.sparse_weight,
            self.rrf_k,
            self.retrieval_multiplier,
        )

    # ==========================================================
    # DOCUMENT ID
    # ==========================================================

    @staticmethod
    def _document_id(
        document: Dict[str, Any],
    ) -> str:
        """
        Return a stable document ID.
        """

        if document.get("id"):
            return str(
                document["id"]
            )

        metadata = document.get(
            "metadata",
            {},
        )

        if isinstance(
            metadata,
            dict,
        ):
            metadata_id = (
                metadata.get(
                    "document_id"
                )
                or metadata.get(
                    "id"
                )
            )

            if metadata_id:
                return str(
                    metadata_id
                )

        content = str(
            document.get(
                "content",
                document.get(
                    "passage",
                    "",
                ),
            )
            or ""
        )

        return hashlib.sha1(
            content.encode(
                "utf-8",
                errors="ignore",
            )
        ).hexdigest()

    # ==========================================================
    # INDEXING
    # ==========================================================

    def index_documents(
        self,
        documents: List[
            Dict[str, Any]
        ],
    ) -> int:
        """
        Index documents into both FAISS and BM25.

        Returns:
            Number of indexed documents.
        """

        if not documents:
            logger.warning(
                "No documents received for indexing."
            )
            return 0

        valid_documents: List[
            Dict[str, Any]
        ] = []

        for index, document in enumerate(
            documents
        ):
            if not isinstance(
                document,
                dict,
            ):
                continue

            content = str(
                document.get(
                    "content",
                    document.get(
                        "passage",
                        "",
                    ),
                )
                or ""
            ).strip()

            if not content:
                continue

            doc = dict(
                document
            )

            if not doc.get("id"):
                doc["id"] = (
                    f"document-{index}"
                )

            doc["content"] = content

            valid_documents.append(
                doc
            )

        if not valid_documents:
            logger.warning(
                "No valid documents available "
                "for indexing."
            )
            return 0

        # ------------------------------------------------------
        # Generate BGE-M3 embeddings
        # ------------------------------------------------------

        passages = [
            document["content"]
            for document in valid_documents
        ]

        logger.info(
            "Generating BGE-M3 embeddings "
            "for %d documents...",
            len(passages),
        )

        embeddings = (
            self.embedding_model
            .encode_passages(
                passages
            )
        )

        # ------------------------------------------------------
        # Validate embedding dimensions
        # ------------------------------------------------------

        expected_dimension = (
            self.embedding_model.dimension
        )

        if embeddings.ndim != 2:
            raise ValueError(
                "Embedding output must be 2-dimensional."
            )

        if embeddings.shape[1] != (
            expected_dimension
        ):
            raise ValueError(
                "Embedding dimension mismatch: "
                f"expected {expected_dimension}, "
                f"received {embeddings.shape[1]}."
            )

        # ------------------------------------------------------
        # FAISS
        # ------------------------------------------------------

        self.faiss_store.add_documents(
            valid_documents,
            embeddings,
        )

        # ------------------------------------------------------
        # BM25
        # ------------------------------------------------------

        self.bm25_store.add_documents(
            valid_documents
        )

        logger.info(
            "Successfully indexed %d documents "
            "into FAISS + BM25.",
            len(valid_documents),
        )

        return len(valid_documents)

    # ==========================================================
    # RRF FUSION
    # ==========================================================

    def _rrf_fuse(
        self,
        dense_results: List[
            Dict[str, Any]
        ],
        sparse_results: List[
            Dict[str, Any]
        ],
    ) -> List[
        Dict[str, Any]
    ]:
        """
        Weighted Reciprocal Rank Fusion.

        score =
            dense_weight / (rrf_k + dense_rank)
            +
            sparse_weight / (rrf_k + sparse_rank)
        """

        scores: Dict[
            str,
            float,
        ] = {}

        documents: Dict[
            str,
            Dict[str, Any],
        ] = {}

        # ------------------------------------------------------
        # Dense results
        # ------------------------------------------------------

        for rank, document in enumerate(
            dense_results,
            start=1,
        ):
            doc_id = (
                self._document_id(
                    document
                )
            )

            documents.setdefault(
                doc_id,
                dict(document),
            )

            scores[doc_id] = (
                scores.get(
                    doc_id,
                    0.0,
                )
                + (
                    self.dense_weight
                    / (
                        self.rrf_k
                        + rank
                    )
                )
            )

            dense_score = (
                document.get(
                    "similarity_score"
                )
            )

            if dense_score is None:
                dense_score = (
                    document.get(
                        "score"
                    )
                )

            if dense_score is not None:
                try:
                    documents[
                        doc_id
                    ][
                        "vector_similarity"
                    ] = float(
                        dense_score
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    pass

        # ------------------------------------------------------
        # Sparse results
        # ------------------------------------------------------

        for rank, document in enumerate(
            sparse_results,
            start=1,
        ):
            doc_id = (
                self._document_id(
                    document
                )
            )

            if doc_id not in documents:
                documents[
                    doc_id
                ] = dict(document)

            scores[doc_id] = (
                scores.get(
                    doc_id,
                    0.0,
                )
                + (
                    self.sparse_weight
                    / (
                        self.rrf_k
                        + rank
                    )
                )
            )

            bm25_score = (
                document.get(
                    "bm25_score"
                )
            )

            if bm25_score is None:
                bm25_score = (
                    document.get(
                        "score"
                    )
                )

            if bm25_score is not None:
                try:
                    documents[
                        doc_id
                    ][
                        "bm25_score"
                    ] = float(
                        bm25_score
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    pass

        # ------------------------------------------------------
        # Sort
        # ------------------------------------------------------

        ranked_ids = sorted(
            scores.keys(),
            key=lambda doc_id: scores[
                doc_id
            ],
            reverse=True,
        )

        results: List[
            Dict[str, Any]
        ] = []

        for rank, doc_id in enumerate(
            ranked_ids,
            start=1,
        ):
            document = dict(
                documents[doc_id]
            )

            document[
                "rrf_score"
            ] = float(
                scores[doc_id]
            )

            document[
                "rrf_rank"
            ] = rank

            results.append(
                document
            )

        return results

    # ==========================================================
    # RETRIEVE
    # ==========================================================

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Execute:

            BGE-M3
              ↓
            FAISS
              +
            BM25
              ↓
            Weighted RRF
              ↓
            BGE Reranker

        Returns a dictionary compatible with the
        RAGOrchestrator.
        """

        start_time = (
            time.perf_counter()
        )

        query = str(
            query or ""
        ).strip()

        top_k = max(
            1,
            int(top_k),
        )

        # ------------------------------------------------------
        # Empty query
        # ------------------------------------------------------

        if not query:
            return {
                "query": "",
                "results": [],
                "latency_ms": 0.0,
                "dense_count": 0,
                "sparse_count": 0,
                "candidate_count": 0,
                "retrieval_breakdown": {
                    "query_tokens": [],
                    "bm25_top_score": 0.0,
                    "vector_top_score": 0.0,
                    "search_method": (
                        "FAISS + BM25 + "
                        "Weighted RRF + "
                        "Multilingual Reranker"
                    ),
                },
            }

        # ------------------------------------------------------
        # Candidate count
        # ------------------------------------------------------

        candidate_k = max(
            top_k
            * self.retrieval_multiplier,
            10,
        )

        # ------------------------------------------------------
        # 1. BGE-M3 dense retrieval
        # ------------------------------------------------------

        dense_results: List[
            Dict[str, Any]
        ] = []

        dense_start = (
            time.perf_counter()
        )

        try:
            query_embedding = (
                self.embedding_model
                .encode_queries(
                    query
                )
            )

            if query_embedding.ndim != 2:
                raise ValueError(
                    "Query embedding must have "
                    "shape [batch, dimension]."
                )

            if query_embedding.shape[
                0
            ] == 0:
                raise ValueError(
                    "Query embedding is empty."
                )

            dense_results = (
                self.faiss_store.search(
                    query_embedding[0],
                    top_k=candidate_k,
                )
                or []
            )

        except Exception as exc:
            logger.exception(
                "FAISS retrieval failed: %s",
                exc,
            )

        dense_latency_ms = (
            time.perf_counter()
            - dense_start
        ) * 1000.0

        # ------------------------------------------------------
        # 2. BM25 sparse retrieval
        # ------------------------------------------------------

        sparse_results: List[
            Dict[str, Any]
        ] = []

        sparse_start = (
            time.perf_counter()
        )

        try:
            sparse_results = (
                self.bm25_store.search(
                    query,
                    top_k=candidate_k,
                )
                or []
            )

        except Exception as exc:
            logger.exception(
                "BM25 retrieval failed: %s",
                exc,
            )

        sparse_latency_ms = (
            time.perf_counter()
            - sparse_start
        ) * 1000.0

        # ------------------------------------------------------
        # 3. Weighted RRF
        # ------------------------------------------------------

        fusion_start = (
            time.perf_counter()
        )

        fused_results = (
            self._rrf_fuse(
                dense_results,
                sparse_results,
            )
        )

        fusion_latency_ms = (
            time.perf_counter()
            - fusion_start
        ) * 1000.0

        # ------------------------------------------------------
        # 4. Multilingual reranking
        # ------------------------------------------------------

        rerank_start = (
            time.perf_counter()
        )

        candidates = (
            fused_results[
                :candidate_k
            ]
        )

        if candidates:

            try:
                final_results = (
                    self.reranker.rerank(
                        query,
                        candidates,
                        top_n=top_k,
                    )
                    or []
                )

            except Exception as exc:

                logger.exception(
                    "Reranking failed: %s",
                    exc,
                )

                # Safe fallback:
                # use RRF ordering.
                final_results = (
                    candidates[
                        :top_k
                    ]
                )

        else:
            final_results = []

        rerank_latency_ms = (
            time.perf_counter()
            - rerank_start
        ) * 1000.0

        # ------------------------------------------------------
        # Ensure final top_k
        # ------------------------------------------------------

        final_results = final_results[
            :top_k
        ]

        # ------------------------------------------------------
        # Score statistics
        # ------------------------------------------------------

        vector_scores: List[
            float
        ] = []

        for document in dense_results:

            score = document.get(
                "similarity_score"
            )

            if score is None:
                score = document.get(
                    "score"
                )

            if score is not None:
                try:
                    vector_scores.append(
                        float(score)
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    pass

        bm25_scores: List[
            float
        ] = []

        for document in sparse_results:

            score = document.get(
                "bm25_score"
            )

            if score is None:
                score = document.get(
                    "score"
                )

            if score is not None:
                try:
                    bm25_scores.append(
                        float(score)
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    pass

        # ------------------------------------------------------
        # Final latency
        # ------------------------------------------------------

        total_latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000.0

        return {
            "query": query,

            "results": final_results,

            "latency_ms": round(
                total_latency_ms,
                2,
            ),

            "dense_count": len(
                dense_results
            ),

            "sparse_count": len(
                sparse_results
            ),

            "candidate_count": len(
                fused_results
            ),

            "retrieval_breakdown": {
                "query_tokens": query.split(),

                "bm25_top_score": (
                    max(bm25_scores)
                    if bm25_scores
                    else 0.0
                ),

                "vector_top_score": (
                    max(vector_scores)
                    if vector_scores
                    else 0.0
                ),

                "search_method": (
                    "FAISS + BM25 + "
                    "Weighted RRF + "
                    "Multilingual Reranker"
                ),

                "dense_weight": (
                    self.dense_weight
                ),

                "sparse_weight": (
                    self.sparse_weight
                ),

                "rrf_k": self.rrf_k,

                "candidate_k": candidate_k,

                "dense_latency_ms": round(
                    dense_latency_ms,
                    2,
                ),

                "sparse_latency_ms": round(
                    sparse_latency_ms,
                    2,
                ),

                "fusion_latency_ms": round(
                    fusion_latency_ms,
                    2,
                ),

                "rerank_latency_ms": round(
                    rerank_latency_ms,
                    2,
                ),
            },
        }

    # ==========================================================
    # CLEAR INDEX
    # ==========================================================

    def clear(self) -> None:
        """
        Clear both FAISS and BM25 stores.
        """

        self.faiss_store.clear()

        # BM25Store does not currently expose clear(),
        # so reset its in-memory state directly.
        self.bm25_store.corpus.clear()
        self.bm25_store.tokenized_corpus.clear()
        self.bm25_store.bm25 = None

        logger.info(
            "Hybrid retrieval indexes cleared."
        )

    # ==========================================================
    # COUNTS
    # ==========================================================

    def count(self) -> int:
        """
        Return the number of indexed chunks.
        """

        return self.faiss_store.count()