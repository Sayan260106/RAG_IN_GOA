"""
Speech-to-Text service wrapper.

Provides a simple interface around SarvamSpeechClient while
keeping the speech implementation isolated from the rest of
the RAG pipeline.
"""

from typing import Optional

from .sarvam import SarvamSpeechClient


class TranscriptionService:
    """
    Wrapper around SarvamSpeechClient.

    The underlying client returns a dictionary containing the
    transcript and STT metadata. This service exposes only the
    transcript string to callers that need a simple interface.
    """

    def __init__(
        self,
        client: Optional[SarvamSpeechClient] = None,
    ):
        self.client = client or SarvamSpeechClient()

    async def get_transcript(
        self,
        audio_data: bytes,
    ) -> str:
        """
        Transcribe audio and return the transcript text.
        """

        if not audio_data:
            return ""

        result = await self.client.transcribe_audio(
            audio_data
        )

        if not isinstance(result, dict):
            return ""

        return str(
            result.get(
                "transcript",
                "",
            )
            or ""
        ).strip()