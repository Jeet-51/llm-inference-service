"""Configuration settings for LLM Inference Service."""
import os
from dataclasses import dataclass

@dataclass
class Settings:
    model_name: str = os.getenv("MODEL_NAME", "TheBloke/Mistral-7B-Instruct-v0.2-AWQ")
    max_model_len: int = int(os.getenv("MAX_MODEL_LEN", "4096"))
    gpu_memory_utilization: float = float(os.getenv("GPU_MEMORY_UTIL", "0.85"))
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    cache_ttl: int = int(os.getenv("CACHE_TTL", "3600"))
    rate_limit_requests: int = int(os.getenv("RATE_LIMIT", "100"))
    rate_limit_window: int = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    default_max_tokens: int = int(os.getenv("DEFAULT_MAX_TOKENS", "256"))
    default_temperature: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))

settings = Settings()
