"""
HHGoa RAG - Chunking Engine
----------------------------

Chunking strategies:
1. Semantic paragraph chunking
2. Sliding-window chunking
3. Sentence-aware chunking
4. Metadata-aware chunking

The output format is designed to work with:
    Loader → Chunker → Embeddings → FAISS/BM25 → RRF → Reranker
"""

import re
from typing import Any, Dict, List, Optional


# ============================================================
# BASE CHUNKER
# ============================================================

class BaseChunker:

    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def split_sentences(text: str) -> List[str]:
    """
    Lightweight multilingual-friendly sentence splitter.

    Handles:
        .
        ?
        !
        ।
        । 
    """

    if not text:
        return []

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?।])\s+",
        text,
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ============================================================
# SEMANTIC CHUNKER
# ============================================================

class SemanticChunker(BaseChunker):
    """
    Paragraph/sentence-aware semantic chunking.

    target_chunk_size is measured approximately in characters.
    """

    def __init__(
        self,
        target_chunk_size: int = 500,
        min_chunk_size: int = 80,
    ):
        self.target_chunk_size = target_chunk_size
        self.min_chunk_size = min_chunk_size

    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:

        metadata = metadata or {}

        if not text or not text.strip():
            return []

        text = text.strip()

        # First try paragraphs.
        sections = re.split(
            r"\n\s*\n+",
            text,
        )

        # If there are no useful paragraphs,
        # fall back to sentences.
        if len(sections) == 1:
            sections = split_sentences(text)

        sections = [
            section.strip()
            for section in sections
            if section.strip()
        ]

        chunks: List[str] = []
        current = ""

        for section in sections:

            # Very large section → split it by sentences.
            if len(section) > self.target_chunk_size:

                if current:
                    chunks.append(current.strip())
                    current = ""

                sentences = split_sentences(section)

                sentence_buffer = ""

                for sentence in sentences:

                    if (
                        sentence_buffer
                        and
                        len(sentence_buffer) + len(sentence) + 1
                        > self.target_chunk_size
                    ):
                        chunks.append(
                            sentence_buffer.strip()
                        )
                        sentence_buffer = sentence

                    else:
                        sentence_buffer = (
                            f"{sentence_buffer} {sentence}"
                        ).strip()

                if sentence_buffer:
                    chunks.append(
                        sentence_buffer.strip()
                    )

                continue

            # Add section to current chunk.
            candidate = (
                f"{current}\n{section}"
                if current
                else section
            )

            if (
                len(candidate)
                <= self.target_chunk_size
            ):
                current = candidate

            else:
                if current:
                    chunks.append(
                        current.strip()
                    )

                current = section

        if current:
            chunks.append(
                current.strip()
            )

        # Do not lose the complete document when
        # chunks are smaller than the minimum size.
        if not chunks and text:
            chunks = [text]

        # Merge tiny trailing chunks with previous chunk.
        merged: List[str] = []

        for chunk in chunks:

            if (
                merged
                and len(chunk) < self.min_chunk_size
            ):
                merged[-1] = (
                    f"{merged[-1]}\n{chunk}"
                ).strip()
            else:
                merged.append(chunk)

        return self._format_chunks(
            merged,
            metadata,
            strategy="semantic",
        )

    @staticmethod
    def _format_chunks(
        chunks: List[str],
        metadata: Dict[str, Any],
        strategy: str,
    ) -> List[Dict[str, Any]]:

        document_id = (
            metadata.get("document_id")
            or metadata.get("id")
            or "document"
        )

        results = []

        for index, content in enumerate(
            chunks,
            start=1,
        ):

            results.append(
                {
                    "id": f"{document_id}-chunk-{index}",
                    "content": content,
                    "strategy": strategy,
                    "chunk_number": index,
                    "source": metadata.get(
                        "source",
                        "MSMARCO-XI / Goa Knowledge Index",
                    ),
                    "language": metadata.get(
                        "language",
                        "en",
                    ),
                    "category": metadata.get(
                        "category",
                        "general_domain",
                    ),
                    "metadata": {
                        **metadata,
                        "char_count": len(content),
                        "word_count": len(
                            content.split()
                        ),
                    },
                }
            )

        return results


# ============================================================
# SLIDING WINDOW CHUNKER
# ============================================================

class SlidingWindowChunker(BaseChunker):
    """
    Word-based sliding window.

    Example:
        window_size = 250
        overlap = 50

    Chunk 1:
        words 0-249

    Chunk 2:
        words 200-449
    """

    def __init__(
        self,
        window_size: int = 250,
        overlap: int = 50,
    ):
        if overlap >= window_size:
            raise ValueError(
                "overlap must be smaller than window_size"
            )

        self.window_size = window_size
        self.overlap = overlap

    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:

        metadata = metadata or {}

        if not text or not text.strip():
            return []

        words = text.split()

        if not words:
            return []

        step = (
            self.window_size
            - self.overlap
        )

        chunks: List[str] = []

        for start in range(
            0,
            len(words),
            step,
        ):

            window = words[
                start:start + self.window_size
            ]

            if not window:
                break

            chunks.append(
                " ".join(window)
            )

            if (
                start + self.window_size
                >= len(words)
            ):
                break

        document_id = (
            metadata.get("document_id")
            or metadata.get("id")
            or "document"
        )

        results = []

        for index, content in enumerate(
            chunks,
            start=1,
        ):

            results.append(
                {
                    "id": (
                        f"{document_id}"
                        f"-chunk-{index}"
                    ),
                    "content": content,
                    "strategy": "sliding_window",
                    "chunk_number": index,
                    "source": metadata.get(
                        "source",
                        "MSMARCO-XI / Goa Knowledge Index",
                    ),
                    "language": metadata.get(
                        "language",
                        "en",
                    ),
                    "category": metadata.get(
                        "category",
                        "general_domain",
                    ),
                    "metadata": {
                        **metadata,
                        "word_count": len(
                            content.split()
                        ),
                    },
                }
            )

        return results


# ============================================================
# SENTENCE-AWARE CHUNKER
# ============================================================

class SentenceAwareChunker(BaseChunker):
    """
    Groups complete sentences until the target
    character size is reached.

    Useful for RAG because retrieved chunks
    contain complete thoughts rather than
    arbitrary character boundaries.
    """

    def __init__(
        self,
        target_chunk_size: int = 500,
        min_chunk_size: int = 80,
    ):
        self.target_chunk_size = target_chunk_size
        self.min_chunk_size = min_chunk_size

    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:

        metadata = metadata or {}

        sentences = split_sentences(text)

        if not sentences:
            return []

        chunks: List[str] = []
        current = ""

        for sentence in sentences:

            candidate = (
                f"{current} {sentence}"
                if current
                else sentence
            )

            if (
                current
                and len(candidate)
                > self.target_chunk_size
            ):
                chunks.append(
                    current.strip()
                )
                current = sentence

            else:
                current = candidate

        if current:
            chunks.append(
                current.strip()
            )

        # Merge very small final chunk.
        if (
            len(chunks) >= 2
            and len(chunks[-1])
            < self.min_chunk_size
        ):
            chunks[-2] = (
                f"{chunks[-2]} "
                f"{chunks[-1]}"
            ).strip()

            chunks.pop()

        document_id = (
            metadata.get("document_id")
            or metadata.get("id")
            or "document"
        )

        results = []

        for index, content in enumerate(
            chunks,
            start=1,
        ):

            results.append(
                {
                    "id": (
                        f"{document_id}"
                        f"-chunk-{index}"
                    ),
                    "content": content,
                    "strategy": "sentence_aware",
                    "chunk_number": index,
                    "source": metadata.get(
                        "source",
                        "MSMARCO-XI / Goa Knowledge Index",
                    ),
                    "language": metadata.get(
                        "language",
                        "en",
                    ),
                    "category": metadata.get(
                        "category",
                        "general_domain",
                    ),
                    "metadata": {
                        **metadata,
                        "char_count": len(content),
                        "sentence_count": len(
                            split_sentences(content)
                        ),
                    },
                }
            )

        return results


# ============================================================
# METADATA-AWARE CHUNKER
# ============================================================

class MetadataAwareChunker(BaseChunker):
    """
    Adds source lineage and structured metadata
    to every chunk.
    """

    def __init__(
        self,
        base_chunker: Optional[BaseChunker] = None,
    ):
        self.base_chunker = (
            base_chunker
            or SentenceAwareChunker()
        )

    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:

        metadata = metadata or {}

        chunks = self.base_chunker.chunk(
            text,
            metadata,
        )

        for chunk in chunks:

            chunk["source"] = metadata.get(
                "source",
                chunk.get(
                    "source",
                    "MSMARCO-XI / Goa Knowledge Index",
                ),
            )

            chunk["language"] = metadata.get(
                "language",
                chunk.get(
                    "language",
                    "en",
                ),
            )

            chunk["category"] = metadata.get(
                "category",
                chunk.get(
                    "category",
                    "general_domain",
                ),
            )

            chunk["metadata"] = {
                **chunk.get("metadata", {}),
                **metadata,
                "indexed_at": "runtime",
            }

        return chunks


# ============================================================
# CHUNKING ROUTER
# ============================================================

class ChunkingRouter:
    """
    Central chunking router.

    Default:
        metadata_aware

    Supported:
        semantic
        sliding_window
        sentence_aware
        metadata_aware
    """

    def __init__(self):

        self.semantic = SemanticChunker()

        self.sliding = SlidingWindowChunker()

        self.sentence_aware = (
            SentenceAwareChunker()
        )

        self.metadata_aware = (
            MetadataAwareChunker(
                self.sentence_aware
            )
        )

    def route_and_chunk(
        self,
        text: str,
        strategy: str = "metadata_aware",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:

        strategy = (
            strategy or "metadata_aware"
        ).lower().strip()

        if strategy == "semantic":

            return self.semantic.chunk(
                text,
                metadata,
            )

        if strategy == "sliding_window":

            return self.sliding.chunk(
                text,
                metadata,
            )

        if strategy == "sentence_aware":

            return self.sentence_aware.chunk(
                text,
                metadata,
            )

        if strategy == "metadata_aware":

            return self.metadata_aware.chunk(
                text,
                metadata,
            )

        raise ValueError(
            f"Unknown chunking strategy: "
            f"{strategy}. "
            f"Supported strategies: "
            f"semantic, sliding_window, "
            f"sentence_aware, metadata_aware"
        )