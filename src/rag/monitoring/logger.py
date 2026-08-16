import json
import logging

class StructuredLogger:
    def __init__(self, name: str = "rag-goa"):
        self.logger = logging.getLogger(name)

    def log_event(self, event: str, **kwargs):
        payload = {"event": event, **kwargs}
        self.logger.info(json.dumps(payload))
