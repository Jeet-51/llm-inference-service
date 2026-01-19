"""Prometheus metrics for monitoring the LLM inference service."""
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

REQUEST_COUNT = Counter('llm_requests_total', 'Total number of inference requests', ['status', 'cached'])
REQUEST_LATENCY = Histogram('llm_request_latency_ms', 'Request latency in milliseconds', buckets=[50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000])
CACHE_HITS = Counter('llm_cache_hits_total', 'Total cache hits')
CACHE_MISSES = Counter('llm_cache_misses_total', 'Total cache misses')
RATE_LIMITED = Counter('llm_rate_limited_total', 'Total rate limited requests')
MODEL_LOADED = Gauge('llm_model_loaded', 'Whether model is loaded (1=yes, 0=no)')
GPU_MEMORY_USED = Gauge('llm_gpu_memory_used_bytes', 'GPU memory used in bytes')

def get_metrics() -> bytes:
    return generate_latest()

def get_metrics_content_type() -> str:
    return CONTENT_TYPE_LATEST
