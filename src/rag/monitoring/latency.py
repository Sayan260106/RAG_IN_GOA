import time
from typing import Dict

class LatencyTracker:
    def __init__(self):
        self.timers: Dict[str, float] = {}
        self.latencies: Dict[str, int] = {}

    def start(self, stage: str):
        self.timers[stage] = time.perf_counter()

    def stop(self, stage: str) -> int:
        if stage in self.timers:
            elapsed = int((time.perf_counter() - self.timers[stage]) * 1000)
            self.latencies[stage] = elapsed
            return elapsed
        return 0
