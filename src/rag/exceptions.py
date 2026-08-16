class RagException(Exception):
    """Base exception for RAG system."""
    pass

class AudioProcessingError(RagException):
    """Raised when audio conversion or ASR fails."""
    pass

class RetrievalError(RagException):
    """Raised when FAISS or BM25 retrieval fails."""
    pass

class GuardrailViolationError(RagException):
    """Raised when confidence or safety guardrails fail."""
    pass
