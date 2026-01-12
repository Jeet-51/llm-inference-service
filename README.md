# LLM Inference Service with Caching and Rate Limiting

A production-grade LLM inference service built with FastAPI and vLLM, optimized for GPU-based serving with AWQ quantization. Designed to demonstrate how large language models can be deployed as reliable, scalable backend infrastructure.

## Overview

This service exposes LLM capabilities through REST APIs while addressing real-world production concerns: latency optimization, concurrent request handling, cost control through caching, and system observability.

Unlike typical LLM demos that treat models as simple API calls, this project applies backend engineering principles to AI workloads, making inference predictable, debuggable, and scalable.

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

**GPU-Optimized Inference**
- vLLM engine with continuous batching for high throughput
- AWQ 4-bit quantization to fit Mistral-7B on T4 (16GB VRAM)
- PagedAttention for efficient memory management

**Caching Layer**
- Redis-based response caching keyed by prompt hash
- Configurable TTL for freshness vs performance tradeoff
- Cache hit/miss metrics for optimization

**Rate Limiting**
- API key-based request throttling
- Sliding window rate limiting via Redis
- Prevents abuse and ensures fair resource allocation

**Observability**
- Prometheus metrics for latency, throughput, and GPU utilization
- Structured logging for debugging
- Health check endpoints for monitoring

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
| GPU | NVIDIA T4 (16GB) |

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
  "latency_ms": 145,
  "cached": false,
  "model": "mistral-7b-instruct-awq"
}
```

## Performance Benchmarks

> **Note**: Benchmarks measured on NVIDIA T4 GPU with Mistral-7B-Instruct-AWQ model.

| Metric | Value |
|--------|-------|
| P50 Latency | `TODO` |
| P95 Latency | `TODO` |
| P99 Latency | `TODO` |
| Throughput | `TODO` req/s |
| Cache Hit Latency | `TODO` |
| GPU Memory Usage | `TODO` |
| Concurrent Users Tested | `TODO` |

*Benchmarks will be updated after load testing with Locust.*

## Project Structure

```
├── app/
│   ├── main.py              # FastAPI application
│   ├── inference.py         # vLLM inference logic
│   ├── cache.py             # Redis caching layer
│   ├── rate_limiter.py      # Rate limiting middleware
│   ├── metrics.py           # Prometheus metrics
│   └── config.py            # Configuration settings
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── locustfile.py            # Load testing script
└── README.md
```

## Running Locally

### Prerequisites
- NVIDIA GPU with CUDA support
- Docker and Docker Compose
- 16GB+ GPU VRAM (T4 or better)

### Quick Start

```bash
# Clone repository
git clone https://github.com/Jeet-51/llm-inference-service.git
cd llm-inference-service

# Start services
docker-compose up --build

# Service available at http://localhost:8000
```

### Running on Google Colab (T4)

```python
# Install dependencies
!pip install vllm fastapi uvicorn redis prometheus-client

# Run the service
!python app/main.py
```

## Load Testing

```bash
# Install Locust
pip install locust

# Run load test
locust -f locustfile.py --host=http://localhost:8000
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `TheBloke/Mistral-7B-Instruct-v0.2-AWQ` | HuggingFace model |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection |
| `RATE_LIMIT` | `100` | Requests per minute per key |
| `CACHE_TTL` | `3600` | Cache expiry in seconds |

## Why This Architecture

**Why vLLM?**
- Continuous batching handles concurrent requests efficiently
- PagedAttention reduces memory fragmentation
- Industry standard for LLM serving (used at scale by major companies)

**Why AWQ Quantization?**
- Reduces model size by 4x with minimal quality loss
- Fits 7B parameter model comfortably on T4 (16GB VRAM)
- Faster inference than FP16

**Why Redis for Caching?**
- Sub-millisecond lookups
- Handles rate limiting and caching in single dependency
- Production-proven at scale

## What This Project Demonstrates

- Deploying LLMs as production backend services
- GPU inference optimization with quantization
- Applying software engineering best practices to AI systems
- Building observable, rate-limited, cached inference pipelines
- Designing for throughput and latency requirements

## Future Improvements

- [ ] Streaming responses via WebSocket
- [ ] Multi-model support
- [ ] Request queuing for traffic spikes
- [ ] Kubernetes deployment manifests
- [ ] Grafana dashboard for metrics

## License

MIT

## Author

Jeet Patel - [LinkedIn](https://linkedin.com/in/pateljeet22) | [GitHub](https://github.com/Jeet-51)
