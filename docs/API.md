# API Documentation

### `POST /api/v1/query`
- **Request Body**:
  ```json
  {
    "query": "What are the primary factors affecting monsoon patterns in North Goa?",
    "include_chunks": true,
    "language": "en"
  }
  ```
- **Response**:
  ```json
  {
    "transcript": "What are the primary factors affecting monsoon patterns in North Goa?",
    "answer": "In North Goa, monsoon patterns are primarily driven by...",
    "confidence": 0.91,
    "latency_ms": 142,
    "grounding_score": 0.96,
    "retrieved_chunks": [...]
  }
  ```
