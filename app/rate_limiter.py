"""Sliding window rate limiter using Redis."""
import time
from typing import Tuple
import redis
from app.config import settings

class RateLimiter:
    def __init__(self):
        self.client = redis.from_url(settings.redis_url, decode_responses=True)
        self.max_requests = settings.rate_limit_requests
        self.window_seconds = settings.rate_limit_window
    
    def is_allowed(self, api_key: str) -> Tuple[bool, dict]:
        now = time.time()
        window_start = now - self.window_seconds
        key = f"ratelimit:{api_key}"
        try:
            pipe = self.client.pipeline()
            pipe.zremrangebyscore(key, 0, window_start)
            pipe.zcard(key)
            pipe.zadd(key, {f"{now}": now})
            pipe.expire(key, self.window_seconds)
            results = pipe.execute()
            request_count = results[1]
            remaining = max(0, self.max_requests - request_count - 1)
            is_allowed = request_count < self.max_requests
            return is_allowed, {"remaining": remaining, "limit": self.max_requests, "reset_in": self.window_seconds, "window": self.window_seconds}
        except redis.RedisError as e:
            return True, {"remaining": -1, "limit": self.max_requests, "error": str(e)}
    
    def reset(self, api_key: str) -> bool:
        try:
            self.client.delete(f"ratelimit:{api_key}")
            return True
        except redis.RedisError:
            return False

_limiter = None

def get_limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter()
    return _limiter
