# Latency Optimization (Budget < 250ms)

| Stage | Target Latency | P95 Latency |
|---|---|---|
| Audio Ingestion & ASR | 70 ms | 95 ms |
| FAISS + BM25 Hybrid Retrieval | 25 ms | 40 ms |
| Guardrail Check | 10 ms | 18 ms |
| Generation First-Token | 35 ms | 55 ms |
| Total Voice-to-UI | **140 ms** | **208 ms** |
