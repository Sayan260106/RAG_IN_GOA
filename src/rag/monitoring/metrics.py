from typing import Dict, Any

class MetricsCollector:
    def __init__(self):
        self.request_count = 0
        self.total_latency_ms = 0

    def record_query(self, latency_ms: int, confidence: float):
        self.request_count += 1
        self.total_latency_ms += latency_ms
