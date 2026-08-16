import pytest
from src.rag.speech.transcription import TranscriptionService

@pytest.mark.asyncio
async def test_voice_transcription():
    svc = TranscriptionService()
    text = await svc.get_transcript(b"audio_bytes")
    assert len(text) > 0
