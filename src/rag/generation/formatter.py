import re
from typing import Dict, Any

class AnswerFormatter:
    def format_speech_friendly(self, answer: str) -> str:
        # Clean markdown artifacts for crisp TTS readability
        cleaned = re.sub(r'[*_#`]', '', answer)
        return cleaned.strip()
