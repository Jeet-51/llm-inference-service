"""Load testing script for LLM Inference Service."""
from locust import HttpUser, task, between
import random


class LLMUser(HttpUser):
    """Simulates users making inference requests."""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    prompts = [
        "What is Python?",
        "Explain machine learning in one sentence.",
        "What is an API?",
        "Define cloud computing.",
        "What is a database?",
        "Explain Docker briefly.",
        "What is Kubernetes?",
        "Define microservices.",
        "What is REST API?",
        "Explain GPU computing.",
    ]
    
    def on_start(self):
        """Set up API key for the user."""
        self.api_key = f"user-{random.randint(1000, 9999)}"
    
    @task(10)
    def inference_request(self):
        """Make inference request - main task."""
        prompt = random.choice(self.prompts)
        self.client.post(
            "/infer",
            json={"prompt": prompt, "max_tokens": 50},
            headers={"X-API-Key": self.api_key}
        )
    
    @task(2)
    def health_check(self):
        """Check health endpoint."""
        self.client.get("/health")
    
    @task(1)
    def metrics_check(self):
        """Check metrics endpoint."""
        self.client.get("/metrics")
