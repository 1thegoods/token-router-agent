"""
LLM Providers — unified interface for Ollama (local) and Fireworks AI (remote).

Both providers expose the same `complete()` method so the router can swap
between them transparently. Token tracking is built in.
"""

import json
import time
import logging
import requests
from dataclasses import dataclass, field
from typing import Optional

from config import AgentConfig
from models import ModelSpec, Provider

logger = logging.getLogger(__name__)


@dataclass
class CompletionResult:
    """Result from a single LLM completion call."""
    text: str
    model_id: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def is_local(self) -> bool:
        return self.provider == Provider.OLLAMA


@dataclass
class TokenTracker:
    """Tracks cumulative token usage across all calls."""
    fireworks_input_tokens: int = 0
    fireworks_output_tokens: int = 0
    local_input_tokens: int = 0
    local_output_tokens: int = 0
    fireworks_calls: int = 0
    local_calls: int = 0

    def record(self, result: CompletionResult):
        if result.is_local:
            self.local_input_tokens += result.input_tokens
            self.local_output_tokens += result.output_tokens
            self.local_calls += 1
        else:
            self.fireworks_input_tokens += result.input_tokens
            self.fireworks_output_tokens += result.output_tokens
            self.fireworks_calls += 1

    @property
    def total_fireworks_tokens(self) -> int:
        return self.fireworks_input_tokens + self.fireworks_output_tokens

    @property
    def total_local_tokens(self) -> int:
        return self.local_input_tokens + self.local_output_tokens

    def summary(self) -> str:
        return (
            f"╔══ Token Usage Summary ══════════════════════════╗\n"
            f"║  Fireworks AI:                                  ║\n"
            f"║    Calls:          {self.fireworks_calls:>8}                    ║\n"
            f"║    Input tokens:   {self.fireworks_input_tokens:>8}                    ║\n"
            f"║    Output tokens:  {self.fireworks_output_tokens:>8}                    ║\n"
            f"║    TOTAL tokens:   {self.total_fireworks_tokens:>8}  ← SCORED         ║\n"
            f"║  Local (Ollama):                                ║\n"
            f"║    Calls:          {self.local_calls:>8}                    ║\n"
            f"║    Input tokens:   {self.local_input_tokens:>8}                    ║\n"
            f"║    Output tokens:  {self.local_output_tokens:>8}                    ║\n"
            f"║    TOTAL tokens:   {self.total_local_tokens:>8}  (free)           ║\n"
            f"╚═════════════════════════════════════════════════╝"
        )


class OllamaProvider:
    """Local Ollama inference — zero Fireworks tokens."""

    def __init__(self, config: AgentConfig):
        self.base_url = config.ollama.base_url
        self.timeout = config.ollama.timeout
        self._available_models: Optional[list[str]] = None

    def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except requests.ConnectionError:
            return False

    def list_models(self) -> list[str]:
        """List locally available Ollama models."""
        if self._available_models is not None:
            return self._available_models
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            data = r.json()
            self._available_models = [m["name"] for m in data.get("models", [])]
            return self._available_models
        except Exception as e:
            logger.warning(f"Failed to list Ollama models: {e}")
            return []

    def has_model(self, model_id: str) -> bool:
        """Check if a specific model is available locally."""
        available = self.list_models()
        # Match by base name (e.g., "llama3.2" matches "llama3.2:latest")
        return any(
            m == model_id or m.startswith(f"{model_id}:")
            for m in available
        )

    def complete(
        self,
        model: ModelSpec,
        messages: list[dict],
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> CompletionResult:
        """Run a completion through Ollama."""
        start = time.perf_counter()

        try:
            # Use the chat API
            payload = {
                "model": model.id,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            }

            r = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()

            latency = (time.perf_counter() - start) * 1000

            # Extract token counts from Ollama response
            eval_count = data.get("eval_count", 0)
            prompt_eval_count = data.get("prompt_eval_count", 0)

            return CompletionResult(
                text=data.get("message", {}).get("content", ""),
                model_id=model.id,
                provider=Provider.OLLAMA,
                input_tokens=prompt_eval_count,
                output_tokens=eval_count,
                latency_ms=latency,
            )

        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            logger.error(f"Ollama completion failed ({model.id}): {e}")
            return CompletionResult(
                text="",
                model_id=model.id,
                provider=Provider.OLLAMA,
                latency_ms=latency,
                success=False,
                error=str(e),
            )


class FireworksProvider:
    """Fireworks AI inference — tokens count toward competition score."""

    def __init__(self, config: AgentConfig):
        self.api_key = config.fireworks.api_key
        self.base_url = config.fireworks.base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)

    def complete(
        self,
        model: ModelSpec,
        messages: list[dict],
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> CompletionResult:
        """Run a completion through the Fireworks API (OpenAI-compatible)."""
        if not self.is_available:
            return CompletionResult(
                text="",
                model_id=model.id,
                provider=Provider.FIREWORKS,
                success=False,
                error="No Fireworks API key configured",
            )

        start = time.perf_counter()

        try:
            payload = {
                "model": model.id,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            r = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=120,
            )
            r.raise_for_status()
            data = r.json()

            latency = (time.perf_counter() - start) * 1000

            choice = data["choices"][0]
            usage = data.get("usage", {})

            return CompletionResult(
                text=choice["message"]["content"],
                model_id=model.id,
                provider=Provider.FIREWORKS,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                latency_ms=latency,
            )

        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            logger.error(f"Fireworks completion failed ({model.id}): {e}")
            return CompletionResult(
                text="",
                model_id=model.id,
                provider=Provider.FIREWORKS,
                latency_ms=latency,
                success=False,
                error=str(e),
            )
