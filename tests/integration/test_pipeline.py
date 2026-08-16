import pytest
from src.rag.pipeline import VoiceRagPipeline

@pytest.mark.asyncio
async def test_pipeline():
    pipeline = VoiceRagPipeline()
    result = await pipeline.process_voice(b"fake_audio")
    assert "transcript" in result
    assert "answer" in result
    assert result["confidence"] > 0
