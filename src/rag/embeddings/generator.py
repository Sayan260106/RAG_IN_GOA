from typing import List, Dict, Any

from .model import MultilingualEmbeddingModel


class EmbeddingGenerator:
    """
    Generates embeddings for RAG chunks using the configured
    multilingual embedding model.

    Default model:
        BAAI/bge-m3

    BGE-M3 produces 1024-dimensional normalized embeddings.
    """

    def __init__(
        self,
        model: MultilingualEmbeddingModel | None = None,
    ):
        self.model = model or MultilingualEmbeddingModel()

    def generate_for_chunks(
        self,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Generate passage embeddings for all chunks.

        Each chunk receives:
            chunk["embedding"] = List[float]
        """

        if not chunks:
            return []

        texts = [
            str(chunk.get("content", "")).strip()
            for chunk in chunks
        ]

        # Generate passage embeddings.
        vectors = self.model.encode_passages(texts)

        for chunk, vector in zip(chunks, vectors):
            chunk["embedding"] = vector.tolist()

        return chunks

    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate an embedding for a user query.

        Returns:
            List[float]
        """

        if not query or not query.strip():
            return []

        vector = self.model.encode_queries(query)

        return vector[0].tolist()