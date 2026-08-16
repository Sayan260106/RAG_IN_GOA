# Retrieval Architecture

- **Dense Store**: FAISS IndexFlatIP with normalized 384-d sentence-transformers embeddings.
- **Sparse Store**: BM25 with tokenization tuned for regional Goan terminology.
- **Fusion**: Reciprocal Rank Fusion ($k=60$) blending sparse exact-match with dense semantic neighborhood.
- **Reranker**: Cross-Encoder MS-MARCO MiniLM.
