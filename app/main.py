"""FastAPI application for LLM inference service."""
from fastapi import FastAPI, HTTPException, Header, Response
from pydantic import BaseModel, Field
from typing import Optional
import time

from app.config import settings
from app.inference import get_engine
from app.cache import get_cache
from app.rate_limiter import get_limiter
from app.metrics import REQUEST_COUNT, REQUEST_LATENCY, CACHE_HITS, CACHE_MISSES, RATE_LIMITED, MODEL_LOADED, get_metrics, get_metrics_content_type

class InferRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4096)
    max_tokens: Optional[int] = Field(default=256, ge=1, le=2048)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)

class InferResponse(BaseModel):
    response: str
    latency_ms: float
    cached: bool
    model: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str

app = FastAPI(title="LLM Inference Service", description="Production-grade LLM inference API with caching and rate limiting", version="1.0.0")

_engine = None
_cache = None
_limiter = None

def get_components():
    global _engine, _cache, _limiter
    if _engine is None:
        _engine = get_engine()
        MODEL_LOADED.set(1)
    if _cache is None:
        _cache = get_cache()
    if _limiter is None:
        _limiter = get_limiter()
    return _engine, _cache, _limiter

@app.post("/infer", response_model=InferResponse)
async def infer(request: InferRequest, x_api_key: str = Header(..., description="API key for authentication")):
    engine, cache, limiter = get_components()
    allowed, limit_info = limiter.is_allowed(x_api_key)
    if not allowed:
        RATE_LIMITED.inc()
        REQUEST_COUNT.labels(status="rate_limited", cached="false").inc()
        raise HTTPException(status_code=429, detail={"error": "Rate limit exceeded", "retry_after": limit_info["reset_in"], "limit": limit_info["limit"]})
    cached_response = cache.get(request.prompt, request.max_tokens, request.temperature)
    if cached_response:
        CACHE_HITS.inc()
        REQUEST_COUNT.labels(status="success", cached="true").inc()
        REQUEST_LATENCY.observe(cached_response["latency_ms"])
        return InferResponse(response=cached_response["response"], latency_ms=cached_response["latency_ms"], cached=True, model=settings.model_name)
    CACHE_MISSES.inc()
    try:
        response_text, latency_ms = engine.generate(prompt=request.prompt, max_tokens=request.max_tokens, temperature=request.temperature)
    except Exception as e:
        REQUEST_COUNT.labels(status="error", cached="false").inc()
        raise HTTPException(status_code=500, detail=str(e))
    cache.set(request.prompt, request.max_tokens, request.temperature, response_text, latency_ms)
    REQUEST_COUNT.labels(status="success", cached="false").inc()
    REQUEST_LATENCY.observe(latency_ms)
    return InferResponse(response=response_text, latency_ms=latency_ms, cached=False, model=settings.model_name)

@app.get("/health", response_model=HealthResponse)
async def health():
    engine_loaded = _engine is not None
    return HealthResponse(status="healthy" if engine_loaded else "starting", model_loaded=engine_loaded, model_name=settings.model_name)

@app.get("/metrics")
async def metrics():
    return Response(content=get_metrics(), media_type=get_metrics_content_type())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
