"""vLLM Inference Engine wrapper."""
import time
from typing import Optional
from vllm import LLM, SamplingParams
from app.config import settings

class InferenceEngine:
    _instance: Optional["InferenceEngine"] = None
    
    def __init__(self):
        self.model: Optional[LLM] = None
        self.model_name = settings.model_name
        self._load_model()
    
    def _load_model(self) -> None:
        print(f"Loading model: {self.model_name}")
        start_time = time.time()
        self.model = LLM(
            model=self.model_name,
            quantization="awq",
            dtype="half",
            max_model_len=settings.max_model_len,
            gpu_memory_utilization=settings.gpu_memory_utilization,
            trust_remote_code=True,
        )
        print(f"Model loaded in {time.time() - start_time:.2f}s")
    
    def generate(self, prompt: str, max_tokens: int = None, temperature: float = None, top_p: float = 0.95, stop: list[str] = None) -> tuple[str, float]:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        max_tokens = max_tokens or settings.default_max_tokens
        temperature = temperature or settings.default_temperature
        sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop=stop or [])
        formatted_prompt = f"[INST] {prompt} [/INST]"
        start_time = time.time()
        outputs = self.model.generate([formatted_prompt], sampling_params)
        latency_ms = (time.time() - start_time) * 1000
        return outputs[0].outputs[0].text.strip(), latency_ms
    
    @classmethod
    def get_instance(cls) -> "InferenceEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

def get_engine() -> InferenceEngine:
    return InferenceEngine.get_instance()
