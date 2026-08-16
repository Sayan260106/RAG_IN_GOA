from functools import lru_cache
from ..pipeline import VoiceRagPipeline

@lru_cache()
def get_pipeline():
    return VoiceRagPipeline()
