# Architecture Overview

The `hhgoa-voice-rag` system connects streaming Indian-accented speech ASR (Sarvam.AI / Web Speech) with a hybrid retrieval engine (FAISS + BM25 + Reciprocal Rank Fusion) and grounded generation (Gemini / LLM) optimized for sub-250ms latency.

```
[Voice Input] ──> [Audio Normalizer] ──> [Sarvam / Web ASR]
                                                │
                                                ▼ (Transcript)
                                       [Hybrid Retriever]
                                        ├── FAISS (Dense)
                                        └── BM25 (Sparse)
                                                │
                                                ▼ (RRF Fusion & Rerank)
                                       [Guardrails & Context]
                                                │
                                                ▼
                                       [Grounded Generation]
                                                │
                                                ▼
                                     [Audio TTS & UI Visuals]
```
