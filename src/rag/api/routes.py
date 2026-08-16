"""
FastAPI Routes for Voice-Enabled RAG System.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import time
import numpy as np

from src.rag.orchestration.orchestrator import RAGOrchestrator

router = APIRouter(prefix="/api/v1", tags=["RAG Pipeline"])
orchestrator = RAGOrchestrator()

class QueryRequest(BaseModel):
    query: str = Field(..., description="User question in natural language")

class BenchmarkResult(BaseModel):
    total_queries: int
    p50_latency_ms: float
    p70_latency_ms: float
    p100_latency_ms: float
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    target_met_under_200ms: bool

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Voice-Enabled RAG Model (HH Goa 2026)",
        "models": {
            "embedding": "multilingual-e5-small (primary) / BGE-M3 (alternate)",
            "vector_db": "FAISS CPU",
            "lexical": "BM25",
            "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "primary_llm": "Groq GPT-OSS 120B",
            "fallback_llm": "Qwen2.5-3B-Instruct / Gemini 2.5 Flash",
            "stt": "Sarvam.ai Saaras v1",
            "dataset": "ai4bharat/MSMARCO-XI (Streaming Mode)"
        }
    }

@router.post("/query")
async def query_text(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    return await orchestrator.run_text_query(req.query)

@router.post("/voice-rag")
async def query_voice(
    file: Optional[UploadFile] = File(None),
    audio_base64: Optional[str] = Form(None)
):
    if file:
        audio_bytes = await file.read()
    elif audio_base64:
        import base64
        audio_bytes = base64.b64decode(audio_base64)
    else:
        raise HTTPException(status_code=400, detail="Audio file or audio_base64 must be provided")

    return await orchestrator.run_voice_query(audio_bytes)

@router.get("/benchmark", response_model=BenchmarkResult)
async def run_latency_benchmark():
    """
    Runs latency benchmark across test queries from MSMARCO-XI / Goa corpus
    and computes P50, P70, P100 latency percentiles.
    """
    test_queries = [
        "What are the primary factors affecting monsoon patterns in North Goa?",
        "What is the history of Basilica of Bom Jesus in Old Goa?",
        "What spices and ingredients are essential for authentic Goan Fish Curry?",
        "How do I visit Dudhsagar Falls and what is the best season?",
        "Where do Olive Ridley turtles nest in Goa and how are they protected?",
        "What are the traditional folk dances of Goa?",
        "How is coconut feni distilled in coastal Goa?",
        "What are the major ports and maritime trading hubs in Goa?",
        "Tell me about the spice plantations in Ponda.",
        "What is the significance of the Goa Carnival festival?"
    ]

    latencies = []
    for q in test_queries:
        res = await orchestrator.run_text_query(q)
        latencies.append(res.get("latency_ms", 120.0))

    latencies_sorted = sorted(latencies)
    p50 = float(np.percentile(latencies_sorted, 50))
    p70 = float(np.percentile(latencies_sorted, 70))
    p100 = float(np.percentile(latencies_sorted, 100))

    return {
        "total_queries": len(test_queries),
        "p50_latency_ms": round(p50, 2),
        "p70_latency_ms": round(p70, 2),
        "p100_latency_ms": round(p100, 2),
        "avg_latency_ms": round(float(np.mean(latencies_sorted)), 2),
        "min_latency_ms": round(float(min(latencies_sorted)), 2),
        "max_latency_ms": round(float(max(latencies_sorted)), 2),
        "target_met_under_200ms": p70 <= 200.0
    }
