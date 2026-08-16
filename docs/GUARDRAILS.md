# Guardrails & Grounding

- **Off-topic Filter**: Enforces Goa geographical, historical, cultural, or travel relevance.
- **Confidence Scorer**: Calculates weighted harmonic mean of retrieval & lexical overlap; queries with confidence < 0.70 trigger a clarification prompt.
- **Hallucination Detection**: Extracts claim triples and cross-checks with retrieved chunks.
