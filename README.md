# hhgoa-voice-rag

Ultra-low latency Voice-Activated RAG (Retrieval-Augmented Generation) system tailored for Goa domain knowledge retrieval.

## System Architecture
- **Voice / Speech**: Sarvam AI & Web Speech ASR integration for streaming speech-to-text.
- **Ingestion & Chunking**: Semantic & sliding-window chunking with Goa heritage/tourism knowledge base.
- **Embeddings & Vector Store**: Dense embeddings + FAISS vector search & BM25 sparse keyword search (Reciprocal Rank Fusion hybrid retrieval).
- **Guardrails**: Confidence thresholding, grounding verification, and hallucination checks.
- **Latency Target**: Sub-250ms voice-to-retrieval pipeline.
