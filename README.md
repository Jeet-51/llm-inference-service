# LLM Inference Service with Caching and Rate Limiting

## Overview

This project is a **production-oriented LLM inference service** designed to expose large language models as a **reliable backend API**, rather than as direct model or SDK calls.

The service provides REST endpoints for common text tasks such as summarization, question answering, and rewriting, while addressing real-world engineering concerns including **latency, concurrency, cost control, and observability**.

The goal of this project is not to build a new model, but to demonstrate how LLMs can be **operated as backend infrastructure** in production systems.

---

## Why I Built This

While working on AI-driven applications and agent-based systems, I noticed that most examples treat LLMs as simple API calls. In real systems, however, the main challenges are not text generation itself, but:

- Handling **concurrent requests**
- Reducing **duplicate inference calls**
- Preventing **uncontrolled traffic spikes**
- Monitoring **latency and failure modes**
- Making AI usage predictable and debuggable

This project was built to explore how LLMs behave when deployed behind a service boundary, similar to any other backend dependency.

---

## How I Got the Idea

The idea came from building multiple AI-powered applications where the same prompts or workflows were repeatedly executed across users and services.

Directly calling an LLM for every request led to:
- Higher latency
- Redundant computation
- Difficulty tracking performance issues

By introducing a dedicated inference service with caching and rate limiting, I was able to centralize LLM usage and apply standard backend engineering practices to AI workloads.

---

## What This Service Does

- Exposes REST APIs for LLM-based text tasks
- Caches repeated requests to avoid unnecessary recomputation
- Applies API key–based rate limiting to control traffic
- Handles requests asynchronously for better throughput
- Provides basic observability for inference latency and errors

This allows downstream applications to treat the LLM as a **stable and scalable backend service**.

---

## Architecture

**Core components:**
- **FastAPI** for API routing and request handling
- **Open-source instruction-tuned LLMs** (e.g., Mistral or LLaMA) for inference
- **Redis** for caching and rate limiting
- **Docker** for containerized deployment
- **Prometheus-compatible metrics** for observability

The architecture is intentionally minimal, focusing on clarity and reliability rather than over-engineering.

---

## Key Features

- **REST API Endpoints**
  - `/infer` for single inference requests
  - `/health` for service health checks

- **Caching**
  - Redis-based cache keyed by prompt and model parameters
  - Configurable TTL to balance freshness and performance

- **Rate Limiting**
  - API key–based request limits
  - Prevents abuse and ensures fair usage

- **Observability**
  - Latency and request metrics
  - Cache hit/miss tracking
  - Structured logging for debugging

---

## Tech Stack

- Python 3.10+
- FastAPI
- Redis
- Open-source LLMs (Mistral / LLaMA)
- Docker & Docker Compose
- Prometheus (optional for metrics)

---

## What This Project Demonstrates

This project demonstrates:
- Treating AI models as **backend services**, not demos
- Applying **software engineering principles** to AI systems
- Designing APIs with performance and reliability in mind
- Avoiding vendor lock-in by using open-source models

It is intended as a **software engineering project with an AI focus**, rather than a research or model-training project.

---

## Running Locally

```bash
docker-compose up --build
