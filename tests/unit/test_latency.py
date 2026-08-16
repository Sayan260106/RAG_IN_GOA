import time
from src.rag.monitoring.latency import LatencyTracker

def test_latency_tracker():
    tracker = LatencyTracker()
    tracker.start("retrieval")
    time.sleep(0.01)
    dur = tracker.stop("retrieval")
    assert dur >= 5
