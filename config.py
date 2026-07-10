"""
Configuration for the Token-Efficient Routing Agent.
Loads environment variables and defines system-wide settings.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FireworksConfig:
    """Fireworks AI API configuration."""
    api_key: str = ""
    base_url: str = "https://api.fireworks.ai/inference/v1"

    def __post_init__(self):
        self.api_key = os.getenv("FIREWORKS_API_KEY", self.api_key)
        self.base_url = os.getenv("FIREWORKS_BASE_URL", self.base_url)

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)


@dataclass
class OllamaConfig:
    """Ollama local model configuration."""
    base_url: str = "http://localhost:11434"
    default_model: str = "llama3.2"
    timeout: int = 120

    def __post_init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", self.base_url)
        self.default_model = os.getenv("OLLAMA_MODEL", self.default_model)


@dataclass
class RouterConfig:
    """Routing strategy configuration."""
    # Prefer local models to minimize Fireworks token spend
    prefer_local: bool = True
    # Confidence threshold — below this, escalate to a stronger model
    confidence_threshold: float = 0.7
    # Max retries on a single task before escalating
    max_retries: int = 1
    # Enable confidence-based escalation
    enable_escalation: bool = True
    # Max tokens to allow for a single completion
    max_completion_tokens: int = 2048
    # Temperature for generation
    temperature: float = 0.1
    
    # --- Committee Routing Settings ---
    enable_committee: bool = True
    committee_size: int = 3
    synthesizer_model: str = "qwen3:8b"


@dataclass
class AgentConfig:
    """Top-level agent configuration."""
    fireworks: FireworksConfig = field(default_factory=FireworksConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    # Verbose logging
    verbose: bool = False
    # Track token usage
    track_tokens: bool = True

    def __post_init__(self):
        self.verbose = os.getenv("VERBOSE", "").lower() in ("1", "true", "yes")


def load_config() -> AgentConfig:
    """Load configuration from environment variables."""
    return AgentConfig()
