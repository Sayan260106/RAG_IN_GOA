"""
FAISS CPU Vector Store
----------------------
Dense vector retrieval using FAISS IndexFlatIP.

Embeddings are expected to be L2-normalized, so inner product
behaves as cosine similarity.

Compatible with:
    BAAI/bge-m3              -> 1024 dimensions
    intfloat/multilingual-e5-small -> 384 dimensions
"""

import logging
from typing import List, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


class FaissCPUStore:
    """
    In-memory CPU FAISS vector store.

    IndexFlatIP is used because the embedding model returns
    normalized vectors.
    """

    def __init__(
        self,
        dimension: int,
        metric: str = "inner_product",
    ):
        if dimension <= 0:
            raise ValueError("FAISS dimension must be greater than 0.")

        self.dimension = dimension
        self.metric = metric

        self.index = None
        self.documents: List[Dict[str, Any]] = []

        self._init_index()

    def _init_index(self):
        """Initialize the FAISS CPU index."""

        try:
            import faiss

            if self.metric == "inner_product":
                self.index = faiss.IndexFlatIP(self.dimension)

            elif self.metric == "l2":
                self.index = faiss.IndexFlatL2(self.dimension)

            else:
                raise ValueError(
                    f"Unsupported FAISS metric: {self.metric}"
                )

            logger.info(
                "FAISS CPU index initialized: dimension=%d, metric=%s",
                self.dimension,
                self.metric,
            )

        except ImportError as exc:
            raise RuntimeError(
                "faiss-cpu is not installed. "
                "Install it using: pip install faiss-cpu"
            ) from exc

        except Exception:
            logger.exception("Failed to initialize FAISS index.")
            raise

    def add_documents(
        self,
        docs: List[Dict[str, Any]],
        embeddings: np.ndarray,
    ) -> None:
        """
        Add documents and their embeddings to FAISS.
        """

        if not docs:
            return

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        # Ensure 2D shape.
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if len(docs) != len(embeddings):
            raise ValueError(
                f"Document/embedding count mismatch: "
                f"{len(docs)} documents vs "
                f"{len(embeddings)} embeddings."
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: "
                f"FAISS expects {self.dimension}, "
                f"received {embeddings.shape[1]}."
            )

        # BGE-M3 / E5 embeddings are normalized before reaching FAISS.
        embeddings = np.ascontiguousarray(
            embeddings,
            dtype=np.float32,
        )

        # Make sure document ordering exactly matches FAISS vector ordering.
        self.documents.extend(
            [doc.copy() for doc in docs]
        )

        self.index.add(embeddings)

        logger.info(
            "Added %d documents to FAISS. Total vectors: %d",
            len(docs),
            self.index.ntotal,
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search for the most similar documents.

        Returns documents containing:
            similarity_score
        """

        if not self.documents:
            return []

        if top_k <= 0:
            return []

        top_k = min(
            int(top_k),
            len(self.documents),
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        # Accept either:
        #     [dimension]
        # or:
        #     [1, dimension]
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.ndim != 2:
            raise ValueError(
                "Query embedding must be a 1D or 2D NumPy array."
            )

        if query_embedding.shape[1] != self.dimension:
            raise ValueError(
                f"Query embedding dimension mismatch: "
                f"expected {self.dimension}, "
                f"received {query_embedding.shape[1]}."
            )

        query_embedding = np.ascontiguousarray(
            query_embedding,
            dtype=np.float32,
        )

        # Search FAISS.
        scores, indices = self.index.search(
            query_embedding,
            top_k,
        )

        results: List[Dict[str, Any]] = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):
            # FAISS can return -1 when insufficient vectors exist.
            if index < 0:
                continue

            if index >= len(self.documents):
                continue

            document = self.documents[index].copy()

            document["similarity_score"] = float(score)

            results.append(document)

        return results

    def count(self) -> int:
        """Return number of indexed documents."""
        return len(self.documents)

    def clear(self) -> None:
        """Clear both FAISS vectors and stored documents."""

        self.documents.clear()

        # Re-create the index so FAISS and documents stay synchronized.
        self._init_index()

        logger.info("FAISS index cleared.")

    def save(self, index_path: str) -> None:
        """
        Save FAISS index to disk.

        Note:
            Document metadata is not saved here.
            This method is optional for your current runtime-indexed setup.
        """

        import faiss

        faiss.write_index(
            self.index,
            index_path,
        )

        logger.info(
            "FAISS index saved to %s",
            index_path,
        )

    def load(self, index_path: str) -> None:
        """
        Load a FAISS index from disk.
        """

        import faiss

        loaded_index = faiss.read_index(index_path)

        if loaded_index.d != self.dimension:
            raise ValueError(
                f"Loaded FAISS index dimension {loaded_index.d} "
                f"does not match expected dimension {self.dimension}."
            )

        self.index = loaded_index

        logger.info(
            "FAISS index loaded from %s",
            index_path,
        )