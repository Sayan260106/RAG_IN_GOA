class RetryPolicy:
    def __init__(self, max_retries: int = 2, backoff_factor: float = 0.5):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
