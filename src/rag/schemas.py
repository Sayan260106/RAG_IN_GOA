from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class RetrievedChunk(BaseModel):
    id: str
    content: str
    source: str
    category: str
    similarity_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

class QueryRequest(BaseModel):
    query: Optional[str] = None
    audio_base64: Optional[str] = None
    language: str = "en"
    include_chunks: bool = True

class QueryResponse(BaseModel):
    transcript: str
    answer: str
    confidence: float
    latency_ms: int
    grounding_score: float
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
