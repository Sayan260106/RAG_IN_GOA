import pytest
from src.rag.guardrails.off_topic import OffTopicGuardrail
from src.rag.guardrails.confidence import ConfidenceScorer

def test_guardrails():
    off_topic = OffTopicGuardrail()
    assert off_topic.is_goa_domain("Tell me about Goa carnival") is True

def test_confidence_scorer():
    scorer = ConfidenceScorer()
    conf = scorer.calculate_confidence([0.9, 0.95], "Goa climate")
    assert 0.0 <= conf <= 1.0
