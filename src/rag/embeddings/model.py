"""
Multilingual Embedding Model
----------------------------
Primary:
    BAAI/bge-m3

Alternative:
    intfloat/multilingual-e5-small

Used by:
    FAISS dense retrieval
    Hybrid BM25 + dense retrieval

BGE-M3:
    - 1024-dimensional embeddings
    - multilingual
    - suitable for Indic languages

Multilingual-E5-small:
    - 384-dimensional embeddings
    - requires query:/passage: prefixes
"""

import logging
from typing import List, Union

import numpy as np

logger = logging.getLogger(__name__)


class MultilingualEmbeddingModel:
    """
    Wrapper around SentenceTransformers for multilingual embeddings.

    The embeddings are L2-normalized so that FAISS IndexFlatIP
    effectively performs cosine similarity.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        normalize_embeddings: bool = True,
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size

        self._model = None

        # Correct embedding dimensions for supported models.
        if "multilingual-e5-small" in model_name.lower():
            self._dimension = 384
        elif "bge-m3" in model_name.lower():
            self._dimension = 1024
        else:
            self._dimension = None

    @property
    def dimension(self) -> int:
        """
        Return embedding dimension.

        For supported models this is known without loading the model.
        For unknown models, the actual dimension is discovered after
        loading the model.
        """
        if self._dimension is None:
            self._load_model()

        return self._dimension

    def _load_model(self):
        """Load SentenceTransformer lazily."""

        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            logger.info(
                "Loading embedding model '%s' on %s",
                self.model_name,
                self.device,
            )

            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )

            # Discover dimension from the actual model.
            if self._dimension is None:
                self._dimension = self._model.get_sentence_embedding_dimension()

            logger.info(
                "Embedding model loaded successfully. Dimension=%s",
                self._dimension,
            )

        except Exception as exc:
            logger.exception(
                "Failed to load embedding model '%s'.",
                self.model_name,
            )

            # Do NOT silently create random embeddings.
            raise RuntimeError(
                f"Could not load embedding model '{self.model_name}'. "
                f"Check your internet connection, Hugging Face access, "
                f"model name, and installed dependencies."
            ) from exc

    def encode_queries(
        self,
        queries: Union[str, List[str]],
    ) -> np.ndarray:
        """
        Encode user queries.

        E5 models require:
            query: <text>

        BGE-M3 does not require the E5 prefix.
        """

        if isinstance(queries, str):
            queries = [queries]

        if not queries:
            return np.empty((0, self.dimension), dtype=np.float32)

        queries = [str(q).strip() for q in queries]

        if "e5" in self.model_name.lower():
            queries = [
                q if q.lower().startswith("query: ")
                else f"query: {q}"
                for q in queries
            ]

        return self._encode(queries)

    def encode_passages(
        self,
        passages: Union[str, List[str]],
    ) -> np.ndarray:
        """
        Encode corpus passages.

        E5 models require:
            passage: <text>

        BGE-M3 does not require the E5 prefix.
        """

        if isinstance(passages, str):
            passages = [passages]

        if not passages:
            return np.empty((0, self.dimension), dtype=np.float32)

        passages = [str(p).strip() for p in passages]

        if "e5" in self.model_name.lower():
            passages = [
                p if p.lower().startswith("passage: ")
                else f"passage: {p}"
                for p in passages
            ]

        return self._encode(passages)

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Generate normalized float32 embeddings."""

        self._load_model()

        try:
            embeddings = self._model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            embeddings = np.asarray(
                embeddings,
                dtype=np.float32,
            )

            # Make sure output is always 2-dimensional.
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)

            # Safety check against FAISS dimension mismatch.
            if embeddings.shape[1] != self.dimension:
                raise ValueError(
                    f"Embedding dimension mismatch: "
                    f"expected {self.dimension}, "
                    f"received {embeddings.shape[1]}"
                )

            # Extra normalization safety.
            if self.normalize_embeddings:
                norms = np.linalg.norm(
                    embeddings,
                    axis=1,
                    keepdims=True,
                )

                embeddings = embeddings / np.maximum(
                    norms,
                    1e-12,
                )

            return np.ascontiguousarray(
                embeddings,
                dtype=np.float32,
            )

        except Exception as exc:
            logger.exception(
                "Embedding generation failed."
            )

            raise RuntimeError(
                f"Failed to generate embeddings using "
                f"'{self.model_name}'."
            ) from exc