# LLM Inference Service with Caching and Rate Limiting

A production-grade LLM inference service built with FastAPI and vLLM, optimized for GPU-based serving with AWQ quantization.

## Overview

This service exposes LLM capabilities through REST APIs while addressing real-world production concerns: latency optimization, concurrent request handling, cost control through caching, and system observability.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Requests                          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Gateway                            │
│              (Request Validation, API Key Auth)                 │
└─────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
        ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
        │ Rate Limiter │ │ Redis Cache  │ │  Prometheus  │
        │   (Redis)    │ │  (Response)  │ │   Metrics    │
        └──────────────┘ └──────────────┘ └──────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      vLLM Inference Engine                      │
│         (Mistral-7B-Instruct + AWQ 4-bit Quantization)         │
│                    Running on NVIDIA T4 GPU                     │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

- **GPU-Optimized Inference**: vLLM with continuous batching and PagedAttention
- **AWQ 4-bit Quantization**: Fits Mistral-7B on T4 GPU (16GB VRAM)
- **Redis Caching**: Sub-5ms response for repeated prompts
- **Rate Limiting**: Sliding window algorithm per API key
- **Prometheus Metrics**: Latency, throughput, cache hit rate
- **Grafana Dashboard**: Real-time monitoring

## Performance Benchmarks

Tested on NVIDIA T4 GPU (Google Colab):

| Metric | Value |
|--------|-------|
| P50 Latency (inference) | 8,273ms |
| P95 Latency (inference) | 8,423ms |
| P99 Latency (inference) | 8,434ms |
| Cache Hit Latency | ~5ms |
| GPU Memory Usage | 14GB / 15GB |
| Cache Hit Rate | 60% |
| Model | Mistral-7B-Instruct-AWQ |

<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/0b65b97f-bdc2-4405-82b4-40c0e6627ebd" />


> Note: Latency on dedicated GPU servers (A10, A100) would be significantly faster.

## Tech Stack

| Component | Technology |
|-----------|------------|
| API Framework | FastAPI |
| Inference Engine | vLLM |
| Model | Mistral-7B-Instruct-v0.2-AWQ |
| Quantization | AWQ (4-bit) |
| Cache & Rate Limiting | Redis |
| Containerization | Docker |
| Metrics | Prometheus |
| Dashboard | Grafana |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/infer` | POST | Run inference with prompt |
| `/health` | GET | Service health check |
| `/metrics` | GET | Prometheus metrics |

### Request Example

```bash
curl -X POST "http://localhost:8000/infer" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"prompt": "Explain microservices in one paragraph", "max_tokens": 256}'
```

### Response Example

```json
{
  "response": "Microservices is an architectural style...",
  "latency_ms": 8273,
  "cached": false,
  "model": "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
}
```

## Project Structure

```
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── inference.py         # vLLM inference logic
│   ├── cache.py             # Redis caching layer
│   ├── rate_limiter.py      # Rate limiting middleware
│   ├── metrics.py           # Prometheus metrics
│   └── config.py            # Configuration settings
├── grafana/
│   └── provisioning/
│       ├── dashboards/
│       │   └── llm-inference.json
│       └── datasources/
│           └── prometheus.yml
├── docker-compose.yml
├── Dockerfile
├── prometheus.yml
├── locustfile.py
├── requirements.txt
├── .env.example
└── README.md
```

## Quick Start

### Prerequisites
- NVIDIA GPU with CUDA support
- Docker and Docker Compose
- 16GB+ GPU VRAM (T4 or better)

### Running with Docker

```bash
# Clone repository
git clone https://github.com/Jeet-51/llm-inference-service.git
cd llm-inference-service

# Start all services
docker-compose up --build

# Services available at:
# - API: http://localhost:8000
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin)
```

### Running on Google Colab (T4)

```python
# Install dependencies
!pip install vllm fastapi uvicorn redis prometheus-client

# Start Redis
!apt-get install redis-server -qq
!redis-server --daemonize yes

# Run the service
!uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Load Testing

```bash
# Install Locust
pip install locust

# Run load test
locust -f locustfile.py --host=http://localhost:8000
```

## Configuration

Environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `TheBloke/Mistral-7B-Instruct-v0.2-AWQ` | HuggingFace model |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection |
| `RATE_LIMIT` | `100` | Requests per minute per key |
| `CACHE_TTL` | `3600` | Cache expiry in seconds |
| `GPU_MEMORY_UTIL` | `0.85` | GPU memory utilization |

## Why This Architecture?

**Why vLLM?**
- Continuous batching handles concurrent requests efficiently
- PagedAttention reduces memory fragmentation
- Industry standard for LLM serving

**Why AWQ Quantization?**
- Reduces model size by 4x with minimal quality loss
- Fits 7B parameter model on T4 (16GB VRAM)
- Faster inference than FP16

**Why Redis for Caching?**
- Sub-millisecond lookups
- Handles rate limiting and caching in single dependency
- Production-proven at scale

## What This Project Demonstrates

- Deploying LLMs as production backend services
- GPU inference optimization with quantization
- Building observable, rate-limited, cached inference pipelines
- Applying software engineering best practices to AI systems
- Designing for throughput and latency requirements

## Future Improvements

- [ ] Streaming responses via WebSocket
- [ ] Multi-model support
- [ ] Request queuing for traffic spikes
- [ ] Kubernetes deployment manifests
- [ ] Authentication with JWT tokens

## License

MIT

## Author

Jeet Patel - [LinkedIn](https://linkedin.com/in/pateljeet22) | [GitHub](https://github.com/Jeet-51)
