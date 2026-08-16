import asyncio

class TimeoutHandler:
    def __init__(self, default_timeout_sec: float = 2.5):
        self.default_timeout_sec = default_timeout_sec

    async def run_with_timeout(self, coro):
        return await asyncio.wait_for(coro, timeout=self.default_timeout_sec)
