"""
Production RAG Orchestrator for HHGoa 2026.

Pipeline:
    input validation
        ↓
    hybrid retrieval
        ↓
    relevance gate
        ↓
    grounded LLM generation
        ↓
    final grounding validation
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

from src.rag.chunking.router import ChunkingRouter
from src.rag.embeddings.model import MultilingualEmbeddingModel
from src.rag.generation.llm import LLMGenerator
from src.rag.guardrails.grounding import GuardrailsManager
from src.rag.ingestion.loader import MSMARCOStreamingLoader
from src.rag.retrieval.hybrid import HybridRetriever
from src.rag.retrieval.reranker import MultilingualReranker
from src.rag.speech.sarvam import SarvamSpeechClient


logger = logging.getLogger(__name__)


class RAGOrchestrator:
    """
    Coordinates the complete HHGoa RAG pipeline.

    Pipeline:
        Dataset
          ↓
        Chunking
          ↓
        BGE-M3 embeddings
          ↓
        FAISS + BM25
          ↓
        Weighted RRF
          ↓
        BGE Reranker
          ↓
        Relevance Gate
          ↓
        Groq / Ollama / configured LLM
          ↓
        Grounding Guardrail
    """

    def __init__(
        self,
        embedding_model_name: Optional[str] = None,
        reranker_model_name: Optional[str] = None,
        latency_budget_ms: float = 200.0,
    ):
        self.latency_budget_ms = float(latency_budget_ms)

        # ---------------------------------------------------------
        # Environment configuration
        # ---------------------------------------------------------

        self.embedding_model_name = (
            embedding_model_name
            or os.getenv(
                "EMBEDDING_MODEL",
                "BAAI/bge-m3",
            )
        )

        self.reranker_model_name = (
            reranker_model_name
            or os.getenv(
                "RERANKER_MODEL",
                "BAAI/bge-reranker-v2-m3",
            )
        )

        # This threshold applies to the BEST relevance score,
        # not the weakest retrieved document.
        self.min_retrieval_similarity = float(
            os.getenv(
                "MIN_RETRIEVAL_SIMILARITY",
                "0.70",
            )
        )

        self.retrieval_top_k = int(
            os.getenv(
                "RETRIEVAL_TOP_K",
                "5",
            )
        )

        self.hf_max_samples = int(
            os.getenv(
                "HF_MAX_SAMPLES",
                "250",
            )
        )

        self.rrf_k = int(
            os.getenv(
                "RRF_K",
                "60",
            )
        )

        self.dense_weight = float(
            os.getenv(
                "DENSE_WEIGHT",
                "0.65",
            )
        )

        self.sparse_weight = float(
            os.getenv(
                "SPARSE_WEIGHT",
                "0.35",
            )
        )

        self.retrieval_multiplier = int(
            os.getenv(
                "RETRIEVAL_MULTIPLIER",
                "4",
            )
        )

        # ---------------------------------------------------------
        # Components
        # ---------------------------------------------------------

        logger.info(
            "Initializing HHGoa RAG Orchestrator..."
        )

        logger.info(
            "Embedding model: %s",
            self.embedding_model_name,
        )

        logger.info(
            "Reranker model: %s",
            self.reranker_model_name,
        )

        self.speech_client = SarvamSpeechClient()

        # ---------------------------------------------------------
        # Embedding model
        # ---------------------------------------------------------

        self.embedding_model = MultilingualEmbeddingModel(
            model_name=self.embedding_model_name
        )

        logger.info(
            "Embedding dimension: %s",
            self.embedding_model.dimension,
        )

        # ---------------------------------------------------------
        # Chunking
        # ---------------------------------------------------------

        self.chunker = ChunkingRouter()

        # ---------------------------------------------------------
        # Reranker
        # ---------------------------------------------------------

        self.reranker = MultilingualReranker(
            model_name=self.reranker_model_name
        )

        # ---------------------------------------------------------
        # Hybrid retrieval
        # ---------------------------------------------------------

        self.retriever = HybridRetriever(
            embedding_model=self.embedding_model,
            reranker=self.reranker,
            rrf_k=self.rrf_k,
            dense_weight=self.dense_weight,
            sparse_weight=self.sparse_weight,
            retrieval_multiplier=self.retrieval_multiplier,
        )

        # ---------------------------------------------------------
        # LLM
        # ---------------------------------------------------------

        self.generator = LLMGenerator()

        # ---------------------------------------------------------
        # Guardrails
        # ---------------------------------------------------------

        self.guardrails = GuardrailsManager(
            min_confidence_threshold=(
                self.min_retrieval_similarity
            )
        )

        # ---------------------------------------------------------
        # Index statistics
        # ---------------------------------------------------------

        self.indexed_documents = 0
        self.indexed_chunks = 0

        # ---------------------------------------------------------
        # Build initial index
        # ---------------------------------------------------------

        self._bootstrap_indices()

        logger.info(
            "HHGoa RAG Orchestrator initialized successfully."
        )

    # =============================================================
    # INDEXING
    # =============================================================

    def _bootstrap_indices(self) -> None:
        """Load dataset, chunk documents and build FAISS + BM25."""

        logger.info("Building RAG indexes...")

        loader = MSMARCOStreamingLoader(
            max_samples=self.hf_max_samples
        )

        docs = list(loader.stream_dataset())

        self.indexed_documents = len(docs)

        logger.info(
            "Loaded %d source documents.",
            self.indexed_documents,
        )

        all_chunks: List[Dict[str, Any]] = []

        for document_index, document in enumerate(docs):

            if not isinstance(document, dict):
                continue

            text = (
                document.get("passage")
                or document.get("content")
                or document.get("query")
                or ""
            )

            text = str(text).strip()

            if not text:
                continue

            metadata = document.get("metadata", {})

            if not isinstance(metadata, dict):
                metadata = {}

            document_id = (
                document.get("id")
                or metadata.get("id")
                or metadata.get("document_id")
                or f"document-{document_index}"
            )

            source = (
                document.get("source")
                or metadata.get("source")
                or "knowledge-base"
            )

            category = (
                metadata.get("category")
                or document.get("category")
                or "knowledge"
            )

            try:
                chunks = self.chunker.route_and_chunk(
                    text=text,
                    strategy="metadata_aware",
                    metadata={
                        "id": str(document_id),
                        "source": str(source),
                        "category": str(category),
                    },
                )

            except Exception as exc:

                logger.exception(
                    "Chunking failed for document %s: %s",
                    document_id,
                    exc,
                )

                continue

            for chunk_index, chunk in enumerate(chunks):

                if not isinstance(chunk, dict):
                    continue

                content = str(
                    chunk.get("content", "") or ""
                ).strip()

                if not content:
                    continue

                chunk_number = chunk.get(
                    "chunk_number",
                    chunk_index + 1,
                )

                all_chunks.append(
                    {
                        "id": (
                            f"{document_id}-"
                            f"{chunk_number}"
                        ),
                        "content": content,
                        "source": str(source),
                        "category": str(category),
                        "chunk_number": chunk_number,
                        "doc_id": str(document_id),
                        "doc_title": (
                            document.get("title")
                            or metadata.get("title")
                        ),
                    }
                )

        if not all_chunks:
            raise RuntimeError(
                "No valid chunks were created. "
                "Check the dataset configuration and loader."
            )

        logger.info(
            "Indexing %d chunks into FAISS + BM25...",
            len(all_chunks),
        )

        self.indexed_chunks = (
            self.retriever.index_documents(
                all_chunks
            )
        )

        logger.info(
            "RAG index ready: %d source documents, %d chunks.",
            self.indexed_documents,
            self.indexed_chunks,
        )

        if self.indexed_chunks == 0:
            raise RuntimeError(
                "No documents were indexed. "
                "Check the dataset configuration."
            )

    # =============================================================
    # SCORE HELPERS
    # =============================================================

    @staticmethod
    def _float_value(
        chunk: Dict[str, Any],
        *keys: str,
    ) -> Optional[float]:

        for key in keys:

            value = chunk.get(key)

            if value is None:
                continue

            try:
                return float(value)

            except (
                TypeError,
                ValueError,
            ):
                continue

        return None

    @classmethod
    def _similarity(
        cls,
        chunk: Dict[str, Any],
    ) -> float:
        """
        Dense/vector similarity.

        This is retained for diagnostics and fallback scoring.
        """

        value = cls._float_value(
            chunk,
            "similarity_score",
            "similarityScore",
            "vector_similarity",
            "vectorSimilarity",
        )

        return value if value is not None else 0.0

    @classmethod
    def _rerank_score(
        cls,
        chunk: Dict[str, Any],
    ) -> Optional[float]:

        return cls._float_value(
            chunk,
            "rerank_score",
            "rerankScore",
        )

    @classmethod
    def _best_relevance_score(
        cls,
        chunks: List[Dict[str, Any]],
    ) -> float:
        """
        Select the strongest relevance signal.

        Priority:
            1. reranker score
            2. vector similarity

        We intentionally DO NOT use the minimum score of
        retrieved chunks because weak secondary results should
        not invalidate an otherwise strongly relevant query.
        """

        rerank_scores = []

        vector_scores = []

        for chunk in chunks:

            rerank = cls._rerank_score(chunk)

            if rerank is not None:
                rerank_scores.append(rerank)

            vector_scores.append(
                cls._similarity(chunk)
            )

        if rerank_scores:
            return max(rerank_scores)

        return max(
            vector_scores,
            default=0.0,
        )

    # =============================================================
    # RESPONSE
    # =============================================================

    def _base_response(
        self,
        query: str,
        answer: str,
        confidence: float,
        chunks: List[Dict[str, Any]],
        latency_ms: float,
        status: str,
        warning: Optional[str],
        provider: str = "none",
        model: str = "none",
        breakdowns: Optional[
            Dict[str, float]
        ] = None,
    ) -> Dict[str, Any]:

        return {
            "query": query,
            "answer": answer,

            "confidence": round(
                float(confidence),
                3,
            ),

            "original_confidence": round(
                float(confidence),
                3,
            ),

            "is_grounded": (
                status
                == "VERIFIED_GROUNDED"
            ),

            "guardrail_status": status,
            "guardrail_warning": warning,

            "latency_ms": round(
                float(latency_ms),
                2,
            ),

            "latency_breakdown": (
                breakdowns or {}
            ),

            "sourceFile": (
                "/src/rag/orchestration/"
                "orchestrator.py"
            ),

            "indexRef": (
                "FAISS_CPU_BM25_RRF"
            ),

            "chunks": chunks,

            "retrievedChunks": chunks,

            "llm_provider": provider,
            "llmProvider": provider,

            "llm_model": model,
            "modelEngine": model,

            "indexed_documents": (
                self.indexed_documents
            ),

            "indexed_chunks": (
                self.indexed_chunks
            ),
        }

    # =============================================================
    # TEXT QUERY
    # =============================================================

    async def run_text_query(
        self,
        query: str,
    ) -> Dict[str, Any]:

        overall_start = time.perf_counter()

        breakdowns: Dict[str, float] = {}

        query = str(query or "").strip()

        # ---------------------------------------------------------
        # Empty query
        # ---------------------------------------------------------

        if not query:

            return self._base_response(
                query="",
                answer="Please provide a question.",
                confidence=0.0,
                chunks=[],
                latency_ms=(
                    time.perf_counter()
                    - overall_start
                ) * 1000,
                status="GUARDRAIL_BLOCKED",
                warning="Empty query.",
                breakdowns=breakdowns,
            )

        # ---------------------------------------------------------
        # Input guardrails
        # ---------------------------------------------------------

        is_safe, error_msg = (
            self.guardrails.validate_input(query)
        )

        if not is_safe:

            return self._base_response(
                query=query,
                answer=(
                    "Query was declined by "
                    f"guardrails: {error_msg}"
                ),
                confidence=0.0,
                chunks=[],
                latency_ms=(
                    time.perf_counter()
                    - overall_start
                ) * 1000,
                status="GUARDRAIL_BLOCKED",
                warning=error_msg,
                breakdowns=breakdowns,
            )

        # ---------------------------------------------------------
        # Retrieval
        # ---------------------------------------------------------

        retrieval_start = time.perf_counter()

        try:

            retrieval_res = self.retriever.retrieve(
                query,
                top_k=self.retrieval_top_k,
            )

        except Exception as exc:

            logger.exception(
                "Retrieval failed."
            )

            breakdowns["retrieval_ms"] = (
                time.perf_counter()
                - retrieval_start
            ) * 1000

            latency = (
                time.perf_counter()
                - overall_start
            ) * 1000

            breakdowns["total_ms"] = latency

            return self._base_response(
                query=query,
                answer=(
                    "I was unable to retrieve "
                    "information from the "
                    "knowledge base."
                ),
                confidence=0.0,
                chunks=[],
                latency_ms=latency,
                status="RETRIEVAL_ERROR",
                warning=str(exc),
                breakdowns=breakdowns,
            )

        breakdowns["retrieval_ms"] = (
            time.perf_counter()
            - retrieval_start
        ) * 1000

        top_chunks = (
            retrieval_res.get("results", [])
            or []
        )

        # ---------------------------------------------------------
        # Normalize chunks
        # ---------------------------------------------------------

        normalized_chunks = []

        for index, chunk in enumerate(top_chunks):

            normalized_chunks.append(
                {
                    "id": chunk.get(
                        "id",
                        f"chk-{index + 1}",
                    ),

                    "chunkNumber": chunk.get(
                        "chunk_number",
                        index + 1,
                    ),

                    "content": chunk.get(
                        "content",
                        "",
                    ),

                    "source": chunk.get(
                        "source",
                        "knowledge-base",
                    ),

                    "category": chunk.get(
                        "category",
                        "general",
                    ),

                    "similarityScore": round(
                        self._similarity(chunk),
                        4,
                    ),

                    "bm25Score": chunk.get(
                        "bm25_score"
                    ),

                    "vectorSimilarity": chunk.get(
                        "vector_similarity"
                    ),

                    "rrfScore": chunk.get(
                        "rrf_score"
                    ),

                    "rerankScore": chunk.get(
                        "rerank_score"
                    ),

                    "keywords": chunk.get(
                        "keywords",
                        [],
                    ),

                    "docTitle": chunk.get(
                        "doc_title"
                    ),
                }
            )

        # ---------------------------------------------------------
        # RELEVANCE SCORE
        # ---------------------------------------------------------

        best_vector_similarity = max(
            (
                self._similarity(chunk)
                for chunk in top_chunks
            ),
            default=0.0,
        )

        best_rerank_score = max(
            (
                score
                for score in (
                    self._rerank_score(chunk)
                    for chunk in top_chunks
                )
                if score is not None
            ),
            default=0.0,
        )

        best_relevance_score = (
            best_rerank_score
            if best_rerank_score > 0.0
            else best_vector_similarity
        )

        # ---------------------------------------------------------
        # Retrieval diagnostics
        # ---------------------------------------------------------

        retrieval_breakdown = (
            retrieval_res.get(
                "retrieval_breakdown",
                {},
            )
            or {}
        )

        retrieval_breakdown = {
            **retrieval_breakdown,

            "total_docs_indexed": (
                self.indexed_documents
            ),

            "total_chunks_indexed": (
                self.indexed_chunks
            ),

            "best_vector_similarity": round(
                best_vector_similarity,
                4,
            ),

            "best_rerank_score": round(
                best_rerank_score,
                4,
            ),

            "best_relevance_score": round(
                best_relevance_score,
                4,
            ),
        }

        # ---------------------------------------------------------
        # HARD RELEVANCE GATE
        # ---------------------------------------------------------

        if (
            not top_chunks
            or best_relevance_score
            < self.min_retrieval_similarity
        ):

            latency = (
                time.perf_counter()
                - overall_start
            ) * 1000

            warning = (
                "Best retrieved relevance score "
                f"({best_relevance_score:.1%}) is below "
                f"the {self.min_retrieval_similarity:.0%} "
                "grounding threshold."
            )

            breakdowns["total_ms"] = latency

            result = self._base_response(
                query=query,
                answer=(
                    "I couldn't find enough relevant "
                    "information in the knowledge base "
                    "to answer that question accurately."
                ),
                confidence=best_relevance_score,
                chunks=normalized_chunks,
                latency_ms=latency,
                status="FLAGGED_LOW_SIMILARITY",
                warning=warning,
                breakdowns=breakdowns,
            )

            result[
                "retrieval_breakdown"
            ] = retrieval_breakdown

            return result

        # ---------------------------------------------------------
        # LLM generation
        # ---------------------------------------------------------

        generation_start = time.perf_counter()

        try:

            gen_res = (
                await self.generator.generate_answer(
                    query,
                    top_chunks,
                )
            )

        except Exception as exc:

            logger.exception(
                "LLM generation failed."
            )

            breakdowns["generation_ms"] = (
                time.perf_counter()
                - generation_start
            ) * 1000

            latency = (
                time.perf_counter()
                - overall_start
            ) * 1000

            breakdowns["total_ms"] = latency

            result = self._base_response(
                query=query,
                answer=(
                    "I was unable to generate "
                    "an answer at this time."
                ),
                confidence=best_relevance_score,
                chunks=normalized_chunks,
                latency_ms=latency,
                status="GENERATION_ERROR",
                warning=str(exc),
                provider="generation_error",
                model="none",
                breakdowns=breakdowns,
            )

            result[
                "retrieval_breakdown"
            ] = retrieval_breakdown

            return result

        breakdowns["generation_ms"] = (
            time.perf_counter()
            - generation_start
        ) * 1000

        raw_answer = str(
            gen_res.get("answer", "")
            or ""
        ).strip()

        # ---------------------------------------------------------
        # Generation unavailable
        # ---------------------------------------------------------

        if not raw_answer:

            latency = (
                time.perf_counter()
                - overall_start
            ) * 1000

            breakdowns["total_ms"] = latency

            result = self._base_response(
                query=query,
                answer=(
                    "I was unable to generate "
                    "a grounded answer from "
                    "the available knowledge base."
                ),
                confidence=best_relevance_score,
                chunks=normalized_chunks,
                latency_ms=latency,
                status="GENERATION_UNAVAILABLE",
                warning=(
                    "No answer was returned by "
                    "the configured LLM providers."
                ),
                provider=gen_res.get(
                    "provider",
                    "none",
                ),
                model=gen_res.get(
                    "model",
                    "none",
                ),
                breakdowns=breakdowns,
            )

            result[
                "retrieval_breakdown"
            ] = retrieval_breakdown

            return result

        # ---------------------------------------------------------
        # Final grounding validation
        # ---------------------------------------------------------

        grounding_start = time.perf_counter()

        grounding_eval = (
            self.guardrails.evaluate_grounding(
                query,
                raw_answer,
                top_chunks,
                raw_similarity=best_relevance_score,
            )
        )

        breakdowns["guardrail_ms"] = (
            time.perf_counter()
            - grounding_start
        ) * 1000

        # ---------------------------------------------------------
        # Grounding result
        # ---------------------------------------------------------

        if not grounding_eval.get(
            "is_grounded",
            False,
        ):

            raw_answer = (
                "I couldn't produce a sufficiently "
                "grounded answer from the available "
                "knowledge base."
            )

            status = "FLAGGED_LOW_GROUNDING"

            warning = grounding_eval.get(
                "reason",
                "Generated answer did not meet "
                "the grounding requirements.",
            )

            provider = "guardrail_abstain"

            model = gen_res.get(
                "model",
                "none",
            )

        else:

            status = "VERIFIED_GROUNDED"

            warning = None

            provider = gen_res.get(
                "provider",
                "unknown",
            )

            model = gen_res.get(
                "model",
                "unknown",
            )

        # ---------------------------------------------------------
        # Final response
        # ---------------------------------------------------------

        total_latency = (
            time.perf_counter()
            - overall_start
        ) * 1000

        breakdowns["total_ms"] = total_latency

        result = self._base_response(
            query=query,
            answer=raw_answer,
            confidence=grounding_eval.get(
                "confidence",
                best_relevance_score,
            ),
            chunks=normalized_chunks,
            latency_ms=total_latency,
            status=status,
            warning=warning,
            provider=provider,
            model=model,
            breakdowns=breakdowns,
        )

        result[
            "retrieval_breakdown"
        ] = retrieval_breakdown

        return result

    # =============================================================
    # VOICE QUERY
    # =============================================================

    async def run_voice_query(
        self,
        audio_bytes: bytes,
    ) -> Dict[str, Any]:

        overall_start = time.perf_counter()

        if not audio_bytes:

            return self._base_response(
                query="",
                answer="No audio data was received.",
                confidence=0.0,
                chunks=[],
                latency_ms=0.0,
                status="VOICE_INPUT_ERROR",
                warning="Empty audio input.",
            )

        # ---------------------------------------------------------
        # Speech-to-text
        # ---------------------------------------------------------

        stt_start = time.perf_counter()

        try:

            stt_res = (
                await self.speech_client.transcribe_audio(
                    audio_bytes
                )
            )

        except Exception as exc:

            logger.exception(
                "Speech transcription failed."
            )

            return self._base_response(
                query="",
                answer=(
                    "I was unable to transcribe "
                    "the audio."
                ),
                confidence=0.0,
                chunks=[],
                latency_ms=(
                    time.perf_counter()
                    - overall_start
                ) * 1000,
                status="STT_ERROR",
                warning=str(exc),
                breakdowns={
                    "stt_ms": (
                        time.perf_counter()
                        - stt_start
                    ) * 1000
                },
            )

        stt_time = (
            time.perf_counter()
            - stt_start
        ) * 1000

        query = str(
            stt_res.get(
                "transcript",
                "",
            )
            or ""
        ).strip()

        # ---------------------------------------------------------
        # Empty transcript
        # ---------------------------------------------------------

        if not query:

            result = self._base_response(
                query="",
                answer=(
                    "I couldn't understand "
                    "the audio. Please try again."
                ),
                confidence=0.0,
                chunks=[],
                latency_ms=(
                    time.perf_counter()
                    - overall_start
                ) * 1000,
                status="STT_EMPTY",
                warning=(
                    "Speech-to-text returned "
                    "an empty transcript."
                ),
                breakdowns={
                    "stt_ms": round(
                        stt_time,
                        2,
                    )
                },
            )

            result["stt_provider"] = stt_res.get(
                "provider",
                "sarvam.ai",
            )

            result["transcript"] = ""

            return result

        # ---------------------------------------------------------
        # RAG query
        # ---------------------------------------------------------

        result = await self.run_text_query(query)

        # ---------------------------------------------------------
        # Add STT telemetry
        # ---------------------------------------------------------

        result.setdefault(
            "latency_breakdown",
            {},
        )

        result["latency_breakdown"]["stt_ms"] = round(
            stt_time,
            2,
        )

        result["stt_provider"] = stt_res.get(
            "provider",
            "sarvam.ai",
        )

        result["transcript"] = query

        result["latency_ms"] = round(
            float(
                result.get(
                    "latency_ms",
                    0.0,
                )
            )
            + stt_time,
            2,
        )

        return result