import os
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.rag.orchestration.orchestrator import RAGOrchestrator


app = FastAPI(
    title="HHGoa Voice RAG API",
    version="2.0.0",
    description="Production-ready Voice RAG API for HHGoa 2026.",
)


# ============================================================
# CORS
# ============================================================

frontend_origins = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ============================================================
# RAG ENGINE
# ============================================================

rag = RAGOrchestrator()


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=4000,
    )


class DocumentRequest(BaseModel):
    title: str = Field(
        ...,
        min_length=1,
        max_length=300,
    )

    category: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    source: str = Field(
        ...,
        min_length=1,
        max_length=500,
    )

    content: str = Field(
        ...,
        min_length=1,
        max_length=100000,
    )


# ============================================================
# HELPERS
# ============================================================

def _normalize_chunk(
    chunk: dict,
    index: int,
) -> dict:
    """
    Convert the internal RAG chunk format into the frontend
    response format.
    """

    return {
        "id": str(
            chunk.get(
                "id",
                f"chunk-{index + 1}",
            )
        ),

        "chunkNumber": chunk.get(
            "chunkNumber",
            chunk.get(
                "chunk_number",
                index + 1,
            ),
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

        "similarityScore": float(
            chunk.get(
                "similarityScore",
                chunk.get(
                    "similarity_score",
                    0.0,
                ),
            )
            or 0.0
        ),

        "bm25Score": chunk.get(
            "bm25Score",
            chunk.get(
                "bm25_score"
            ),
        ),

        "vectorSimilarity": chunk.get(
            "vectorSimilarity",
            chunk.get(
                "vector_similarity"
            ),
        ),

        "rrfScore": chunk.get(
            "rrfScore",
            chunk.get(
                "rrf_score"
            ),
        ),

        "rerankScore": chunk.get(
            "rerankScore",
            chunk.get(
                "rerank_score"
            ),
        ),

        "keywords": chunk.get(
            "keywords",
            [],
        ),

        "docTitle": chunk.get(
            "docTitle",
            chunk.get(
                "doc_title"
            ),
        ),
    }


def _api_response(
    result: dict,
    query: str,
) -> dict:
    """
    Convert the internal RAG response to the stable API
    contract consumed by the React frontend.
    """

    chunks = [
        _normalize_chunk(
            chunk,
            index,
        )
        for index, chunk in enumerate(
            result.get(
                "chunks",
                [],
            )
        )
        if isinstance(
            chunk,
            dict,
        )
    ]

    breakdown = result.get(
        "retrieval_breakdown",
        {},
    ) or {}

    min_similarity = min(
        (
            chunk["similarityScore"]
            for chunk in chunks
        ),
        default=0.0,
    )

    is_grounded = bool(
        result.get(
            "is_grounded",
            False,
        )
    )

    status = result.get(
        "guardrail_status",
        (
            "VERIFIED_GROUNDED"
            if is_grounded
            else "FLAGGED_LOW_SIMILARITY"
        ),
    )

    return {
        # --------------------------------------------------------
        # Query
        # --------------------------------------------------------

        "query": result.get(
            "query",
            query,
        ),

        "transcript": result.get(
            "transcript",
            query,
        ),

        # --------------------------------------------------------
        # Answer
        # --------------------------------------------------------

        "answer": result.get(
            "answer",
            "",
        ),

        # --------------------------------------------------------
        # Confidence
        # --------------------------------------------------------

        "confidence": float(
            result.get(
                "confidence",
                0.0,
            )
            or 0.0
        ),

        "originalConfidence": float(
            result.get(
                "original_confidence",
                result.get(
                    "confidence",
                    0.0,
                ),
            )
            or 0.0
        ),

        "groundingScore": float(
            result.get(
                "confidence",
                0.0,
            )
            or 0.0
        ),

        # --------------------------------------------------------
        # Latency
        # --------------------------------------------------------

        "latencyMs": int(
            result.get(
                "latency_ms",
                0,
            )
            or 0
        ),

        "latencyBreakdown": result.get(
            "latency_breakdown",
            {},
        ) or {},

        # --------------------------------------------------------
        # Retrieval
        # --------------------------------------------------------

        "sourceFile": result.get(
            "sourceFile",
            "/src/rag/orchestration/orchestrator.py",
        ),

        "indexRef": result.get(
            "indexRef",
            "FAISS_CPU_BM25_RRF",
        ),

        "retrievedChunks": chunks,

        "chunks": chunks,

        "minSimilarity": min_similarity,

        # --------------------------------------------------------
        # LLM
        # --------------------------------------------------------

        "modelEngine": result.get(
            "llm_model",
            "none",
        ),

        "llmProvider": result.get(
            "llm_provider",
            "none",
        ),

        # --------------------------------------------------------
        # Guardrails
        # --------------------------------------------------------

        "isGrounded": is_grounded,

        "guardrailStatus": status,

        "guardrailWarning": result.get(
            "guardrail_warning"
        ),

        # --------------------------------------------------------
        # Retrieval diagnostics
        # --------------------------------------------------------

        "retrievalBreakdown": {
            "queryTokens": breakdown.get(
                "query_tokens",
                [],
            ),

            "bm25TopScore": float(
                breakdown.get(
                    "bm25_top_score",
                    0.0,
                )
                or 0.0
            ),

            "vectorTopScore": float(
                breakdown.get(
                    "vector_top_score",
                    0.0,
                )
                or 0.0
            ),

            "totalDocsIndexed": int(
                breakdown.get(
                    "total_docs_indexed",
                    result.get(
                        "indexed_documents",
                        0,
                    ),
                )
                or 0
            ),

            "totalChunksIndexed": int(
                breakdown.get(
                    "total_chunks_indexed",
                    result.get(
                        "indexed_chunks",
                        0,
                    ),
                )
                or 0
            ),

            "searchMethod": breakdown.get(
                "search_method",
                (
                    "FAISS + BM25 + Weighted RRF "
                    "+ Multilingual Reranker"
                ),
            ),
        },

        # --------------------------------------------------------
        # Statistics
        # --------------------------------------------------------

        "stats": {
            "indexedDocuments": result.get(
                "indexed_documents",
                0,
            ),

            "indexedChunks": result.get(
                "indexed_chunks",
                0,
            ),
        },

        # --------------------------------------------------------
        # Voice metadata
        # --------------------------------------------------------

        "sttProvider": result.get(
            "stt_provider"
        ),

        # --------------------------------------------------------
        # Timestamp
        # --------------------------------------------------------

        "timestamp": datetime.now().isoformat(),
    }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "HHGoa Voice RAG API",
        "version": "2.0.0",
        "docs": "/docs",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "HHGoa Voice RAG",

        "indexedDocuments": (
            rag.indexed_documents
        ),

        "indexedChunks": (
            rag.indexed_chunks
        ),

        "embeddingModel": (
            rag.embedding_model_name
        ),

        "rerankerModel": (
            rag.reranker_model_name
        ),

        "embeddingDimension": (
            rag.embedding_model.dimension
        ),

        "llmPrimary": (
            "groq"
            if rag.generator.groq_api_key
            else "not-configured"
        ),

        "llmFallback": (
            f"ollama/"
            f"{rag.generator.fallback_model}"
        ),
    }


# ============================================================
# TEXT RAG
# ============================================================

@app.post("/api/rag")
async def rag_query(
    request: QueryRequest,
):
    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=422,
            detail="Query cannot be empty.",
        )

    try:
        result = await rag.run_text_query(
            query
        )

        return _api_response(
            result,
            query,
        )

    except Exception:
        # Never return a fake successful answer.
        raise HTTPException(
            status_code=500,
            detail=(
                "RAG processing failed. "
                "Check the backend logs."
            ),
        )


# ============================================================
# QUERY ALIAS
# ============================================================

@app.post("/api/query")
async def query(
    request: QueryRequest,
):
    return await rag_query(
        request
    )


# ============================================================
# DOCUMENTS
# ============================================================

@app.get("/api/documents")
async def get_documents():
    """
    Return the documents/chunks currently loaded into FAISS.

    Runtime document mutation is intentionally disabled because
    the current architecture does not provide persistent index
    rebuilding.
    """

    documents = getattr(
        rag.retriever.faiss_store,
        "documents",
        [],
    )

    return {
        "count": len(documents),

        "totalChunks": len(
            documents
        ),

        "documents": [
            {
                "id": str(
                    document.get(
                        "id",
                        "",
                    )
                ),

                "title": (
                    document.get(
                        "docTitle"
                    )
                    or document.get(
                        "title"
                    )
                    or document.get(
                        "id",
                        "Untitled",
                    )
                ),

                "category": document.get(
                    "category",
                    "general",
                ),

                "source": document.get(
                    "source",
                    "knowledge-base",
                ),

                "content": document.get(
                    "content",
                    "",
                ),

                "chunkCount": 1,

                "createdAt": datetime.now().isoformat(),

                "isCustom": False,
            }

            for document in documents

            if isinstance(
                document,
                dict,
            )
        ],
    }


@app.post("/api/documents")
async def add_document(
    request: DocumentRequest,
):
    """
    Runtime document ingestion is not supported by the current
    persistent-index architecture.

    We explicitly report failure instead of pretending that the
    document was indexed.
    """

    return {
        "success": False,

        "message": (
            "Runtime document ingestion is currently disabled. "
            "Add the document to the configured corpus and "
            "rebuild the RAG indexes."
        ),
    }


@app.post("/api/documents/reset")
async def reset_documents():
    """
    Runtime reset is intentionally disabled.
    """

    return {
        "success": False,

        "message": (
            "Runtime corpus reset is currently disabled. "
            "Rebuild the configured corpus/index to reset it."
        ),
    }


# ============================================================
# BENCHMARK
# ============================================================

@app.get("/api/benchmark")
async def benchmark():
    """
    Return benchmark telemetry.

    The actual benchmark runner is kept separate from the
    request path so that opening the UI does not trigger a
    potentially expensive benchmark.
    """

    return {
        "recent_queries": [],

        "stats": {
            "p50": 0,
            "p75": 0,
            "p100": 0,
            "avg": 0,
            "min": 0,
            "max": 0,
            "total": 0,

            "budgetLimitMs": (
                rag.latency_budget_ms
            ),

            "underBudgetRatio": 0,
        },
    }


# ============================================================
# VOICE RAG
# ============================================================

@app.post("/api/voice")
async def voice_query(
    file: UploadFile = File(...),
):
    """
    Voice → Sarvam STT → RAG → grounded answer.
    """

    audio = await file.read()

    if not audio:
        raise HTTPException(
            status_code=400,
            detail="Audio file is empty.",
        )

    try:

        result = await rag.run_voice_query(
            audio
        )

        return _api_response(
            result,
            result.get(
                "transcript",
                "",
            ),
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail=(
                "Voice RAG processing failed. "
                "Check the backend logs."
            ),
        )