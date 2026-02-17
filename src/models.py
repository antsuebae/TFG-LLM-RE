"""
Gestores de modelos LLM (Ollama, OpenAI y NVIDIA NIM)
"""

import os
import time
import logging
import ollama
from openai import OpenAI
from typing import Optional

logger = logging.getLogger(__name__)

# Modelos disponibles en NVIDIA NIM (build.nvidia.com)
NIM_MODELS = {
    "llama-3.1-70b": "meta/llama-3.1-70b-instruct",
    "llama-3.1-8b": "meta/llama-3.1-8b-instruct",
    "mistral-7b": "mistralai/mistral-7b-instruct-v0.3",
    "mixtral-8x7b": "mistralai/mixtral-8x7b-instruct-v0.1",
}

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds


def _retry_with_backoff(func, max_retries=MAX_RETRIES):
    """Execute func with exponential backoff on failure."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                logger.error(f"All {max_retries} attempts failed. Last error: {e}")
    raise last_error


class OllamaModel:
    """Wrapper para modelos locales via Ollama."""

    def __init__(self, model_name: str, temperature: float = 0.4):
        self.model_name = model_name
        self.temperature = temperature

    def generate(self, prompt: str, max_tokens: int = 256) -> dict:
        """Genera respuesta del modelo."""
        start_time = time.time()

        try:
            def _call():
                return ollama.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    options={
                        "temperature": self.temperature,
                        "num_predict": max_tokens,
                    }
                )

            response = _retry_with_backoff(_call)
            elapsed = time.time() - start_time

            content = response["message"]["content"]
            # Estimate tokens from response (rough: 1 token ~ 4 chars)
            eval_count = response.get("eval_count", len(content) // 4)
            tokens_per_second = eval_count / elapsed if elapsed > 0 else 0

            return {
                "content": content,
                "model": self.model_name,
                "time_seconds": elapsed,
                "tokens_per_second": round(tokens_per_second, 1),
                "success": True,
                "error": None
            }
        except Exception as e:
            logger.error(f"OllamaModel error ({self.model_name}): {e}")
            return {
                "content": "",
                "model": self.model_name,
                "time_seconds": time.time() - start_time,
                "tokens_per_second": 0,
                "success": False,
                "error": str(e)
            }


class OpenAIModel:
    """Wrapper para GPT-3.5 via OpenAI API."""

    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.4, api_key: Optional[str] = None):
        self.model_name = model_name
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()

    def generate(self, prompt: str, max_tokens: int = 256) -> dict:
        """Genera respuesta del modelo."""
        start_time = time.time()

        try:
            def _call():
                return self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                )

            response = _retry_with_backoff(_call)
            elapsed = time.time() - start_time

            tokens_used = response.usage.total_tokens if response.usage else 0
            tokens_per_second = tokens_used / elapsed if elapsed > 0 else 0

            return {
                "content": response.choices[0].message.content,
                "model": self.model_name,
                "time_seconds": elapsed,
                "tokens_used": tokens_used,
                "tokens_per_second": round(tokens_per_second, 1),
                "success": True,
                "error": None
            }
        except Exception as e:
            logger.error(f"OpenAIModel error ({self.model_name}): {e}")
            return {
                "content": "",
                "model": self.model_name,
                "time_seconds": time.time() - start_time,
                "tokens_used": 0,
                "tokens_per_second": 0,
                "success": False,
                "error": str(e)
            }


class NVIDIANIMModel:
    """Wrapper para modelos via NVIDIA NIM API."""

    NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"

    def __init__(self, model_name: str = "llama-3.1-70b", temperature: float = 0.4, api_key: Optional[str] = None):
        self.model_name = NIM_MODELS.get(model_name, model_name)
        self.temperature = temperature
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")

        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY no configurada. Obtenla en https://build.nvidia.com")

        self.client = OpenAI(
            base_url=self.NIM_BASE_URL,
            api_key=self.api_key
        )

    def generate(self, prompt: str, max_tokens: int = 1024) -> dict:
        """Genera respuesta del modelo."""
        start_time = time.time()

        try:
            def _call():
                return self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                )

            response = _retry_with_backoff(_call)
            elapsed = time.time() - start_time

            tokens_used = response.usage.total_tokens if response.usage else 0
            tokens_per_second = tokens_used / elapsed if elapsed > 0 else 0

            return {
                "content": response.choices[0].message.content,
                "model": self.model_name,
                "time_seconds": elapsed,
                "tokens_used": tokens_used,
                "tokens_per_second": round(tokens_per_second, 1),
                "success": True,
                "error": None
            }
        except Exception as e:
            logger.error(f"NVIDIANIMModel error ({self.model_name}): {e}")
            return {
                "content": "",
                "model": self.model_name,
                "time_seconds": time.time() - start_time,
                "tokens_used": 0,
                "tokens_per_second": 0,
                "success": False,
                "error": str(e)
            }


def get_model(model_config: dict, temperature: float = 0.4):
    """Factory para crear instancias de modelos."""
    model_type = model_config.get("type", "ollama")
    model_name = model_config["name"]

    if model_type == "ollama":
        return OllamaModel(model_name, temperature)
    elif model_type == "openai":
        return OpenAIModel(model_name, temperature)
    elif model_type == "nvidia_nim":
        return NVIDIANIMModel(model_name, temperature)
    else:
        raise ValueError(f"Tipo de modelo desconocido: {model_type}")
