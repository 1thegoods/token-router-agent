"""
Model Registry — defines all available models, their costs, and capability tiers.

Fireworks AI pricing is per-token. We rank models into tiers so the router
can pick the cheapest one that's likely to succeed for a given task complexity.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class Tier(IntEnum):
    """Model capability tier. Higher = more capable but more expensive."""
    LOCAL = 0      # Ollama — zero Fireworks tokens
    TINY = 1       # Very small / fast models
    SMALL = 2      # 7-8B class models
    MEDIUM = 3     # 70B class models
    LARGE = 4      # 405B+ / frontier models


class Provider(str):
    OLLAMA = "ollama"
    FIREWORKS = "fireworks"


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a single model endpoint."""
    id: str                         # Model identifier (API model name)
    name: str                       # Human-readable name
    provider: str                   # "ollama" or "fireworks"
    tier: Tier                      # Capability tier
    cost_per_million_input: float   # USD per 1M input tokens (0 for local)
    cost_per_million_output: float  # USD per 1M output tokens (0 for local)
    context_window: int = 8192      # Max context length
    supports_json: bool = True      # Supports JSON mode
    supports_tools: bool = False    # Supports tool/function calling

    @property
    def is_local(self) -> bool:
        return self.provider == Provider.OLLAMA

    @property
    def cost_score(self) -> float:
        """Relative cost score for ranking. Lower = cheaper."""
        return self.cost_per_million_input + self.cost_per_million_output


# ─── Local Ollama Models ────────────────────────────────────────────────────

OLLAMA_MODELS = [
    ModelSpec(
        id="llama3.2",
        name="Llama 3.2 3B (Local)",
        provider=Provider.OLLAMA,
        tier=Tier.LOCAL,
        cost_per_million_input=0.0,
        cost_per_million_output=0.0,
        context_window=131072,
    ),
    ModelSpec(
        id="qwen3.5:4b",
        name="Qwen 3.5 4B (Local)",
        provider=Provider.OLLAMA,
        tier=Tier.LOCAL,
        cost_per_million_input=0.0,
        cost_per_million_output=0.0,
        context_window=32768,
    ),
    ModelSpec(
        id="gemma4:e2b",
        name="Gemma 4 e2b (Local)",
        provider=Provider.OLLAMA,
        tier=Tier.LOCAL,
        cost_per_million_input=0.0,
        cost_per_million_output=0.0,
        context_window=8192,
    ),
    ModelSpec(
        id="qwen3:8b",
        name="Qwen 3 8B (Local)",
        provider=Provider.OLLAMA,
        tier=Tier.LOCAL,
        cost_per_million_input=0.0,
        cost_per_million_output=0.0,
        context_window=32768,
    ),
    ModelSpec(
        id="deepseek-r1:8b",
        name="DeepSeek R1 8B (Local)",
        provider=Provider.OLLAMA,
        tier=Tier.LOCAL,
        cost_per_million_input=0.0,
        cost_per_million_output=0.0,
        context_window=131072,
    ),
]


# ─── Fireworks AI Models (sorted cheapest → most expensive) ─────────────────

FIREWORKS_MODELS = [
    ModelSpec(
        id="accounts/fireworks/models/llama-v3p2-3b-instruct",
        name="Llama 3.2 3B (Fireworks)",
        provider=Provider.FIREWORKS,
        tier=Tier.TINY,
        cost_per_million_input=0.10,
        cost_per_million_output=0.10,
        context_window=131072,
    ),
    ModelSpec(
        id="accounts/fireworks/models/llama-v3p1-8b-instruct",
        name="Llama 3.1 8B (Fireworks)",
        provider=Provider.FIREWORKS,
        tier=Tier.SMALL,
        cost_per_million_input=0.20,
        cost_per_million_output=0.20,
        context_window=131072,
    ),
    ModelSpec(
        id="accounts/fireworks/models/qwen2p5-72b-instruct",
        name="Qwen 2.5 72B (Fireworks)",
        provider=Provider.FIREWORKS,
        tier=Tier.MEDIUM,
        cost_per_million_input=0.90,
        cost_per_million_output=0.90,
        context_window=32768,
    ),
    ModelSpec(
        id="accounts/fireworks/models/llama-v3p1-70b-instruct",
        name="Llama 3.1 70B (Fireworks)",
        provider=Provider.FIREWORKS,
        tier=Tier.MEDIUM,
        cost_per_million_input=0.90,
        cost_per_million_output=0.90,
        context_window=131072,
    ),
    ModelSpec(
        id="accounts/fireworks/models/llama-v3p1-405b-instruct",
        name="Llama 3.1 405B (Fireworks)",
        provider=Provider.FIREWORKS,
        tier=Tier.LARGE,
        cost_per_million_input=3.00,
        cost_per_million_output=3.00,
        context_window=131072,
    ),
    ModelSpec(
        id="accounts/fireworks/models/deepseek-v3",
        name="DeepSeek V3 (Fireworks)",
        provider=Provider.FIREWORKS,
        tier=Tier.LARGE,
        cost_per_million_input=0.90,
        cost_per_million_output=0.90,
        context_window=131072,
        supports_tools=True,
    ),
]


# ─── Combined Registry ──────────────────────────────────────────────────────

ALL_MODELS = OLLAMA_MODELS + FIREWORKS_MODELS


def get_models_by_tier(tier: Tier) -> list[ModelSpec]:
    """Get all models at a specific tier, sorted by cost."""
    return sorted(
        [m for m in ALL_MODELS if m.tier == tier],
        key=lambda m: m.cost_score,
    )


def get_cheapest_fireworks(min_tier: Tier = Tier.TINY) -> Optional[ModelSpec]:
    """Get the cheapest Fireworks model at or above a minimum tier."""
    candidates = sorted(
        [m for m in FIREWORKS_MODELS if m.tier >= min_tier],
        key=lambda m: (m.cost_score, m.tier),
    )
    return candidates[0] if candidates else None


def get_local_models() -> list[ModelSpec]:
    """Get all local (Ollama) models."""
    return [m for m in ALL_MODELS if m.is_local]


def get_model_by_id(model_id: str) -> Optional[ModelSpec]:
    """Look up a model by its ID."""
    for m in ALL_MODELS:
        if m.id == model_id:
            return m
    return None


def get_escalation_chain() -> list[ModelSpec]:
    """
    Returns models in escalation order: local first, then cheapest Fireworks
    to most expensive. This is the order the router tries when escalating.
    """
    local = sorted(get_local_models(), key=lambda m: m.cost_score)
    fireworks = sorted(FIREWORKS_MODELS, key=lambda m: (m.cost_score, m.tier))
    return local + fireworks
