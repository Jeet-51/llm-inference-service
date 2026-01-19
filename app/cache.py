"""Redis caching layer for LLM responses."""
import hashlib
import json
from typing import Optional
import redis
from app.config import settings

class ResponseCache:
    def __init__(self):
        self.client = redis.from_url(settings.redis_url, decode_responses=True)
        self.ttl = settings.cache_ttl
        self.hits = 0
        self.misses = 0
    
    def _hash_prompt(self, prompt: str, max_tokens: int, temperature: float) -> str:
        key_data = f"{prompt}:{max_tokens}:{temperature}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]
    
    def get(self, prompt: str, max_tokens: int, temperature: float) -> Optional[dict]:
        key = self._hash_prompt(prompt, max_tokens, temperature)
        try:
            cached = self.client.get(f"llm:{key}")
            if cached:
                self.hits += 1
                return json.loads(cached)
            self.misses += 1
            return None
        except redis.RedisError as e:
            self.misses += 1
            return None
    
    def set(self, prompt: str, max_tokens: int, temperature: float, response: str, latency_ms: float) -> bool:
        key = self._hash_prompt(prompt, max_tokens, temperature)
        data = {"response": response, "latency_ms": latency_ms}
        try:
            self.client.setex(f"llm:{key}", self.ttl, json.dumps(data))
            return True
        except redis.RedisError:
            return False
    
    def get_stats(self) -> dict:
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {"hits": self.hits, "misses": self.misses, "hit_rate": f"{hit_rate:.1f}%"}

_cache: Optional[ResponseCache] = None

def get_cache() -> ResponseCache:
    global _cache
    if _cache is None:
        _cache = ResponseCache()
    return _cache
