"""
Router — the brain of the agent.

Takes a classified task and decides which model to route it to, then executes
the completion with escalation logic if the first attempt isn't good enough.

Routing Strategy (priority order):
  1. LOCAL first — always try Ollama (zero Fireworks tokens)
  2. CHEAP Fireworks — only if local fails or isn't available
  3. ESCALATE — if cheap model output looks low-confidence, try a bigger model
  4. EXPENSIVE Fireworks — last resort for genuinely hard tasks
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional

from config import AgentConfig
from models import (
    ModelSpec, Tier, Provider,
    get_local_models, get_cheapest_fireworks, get_escalation_chain,
    FIREWORKS_MODELS,
)
from classifier import TaskProfile, Difficulty, TaskType
from providers import (
    OllamaProvider, FireworksProvider, CompletionResult, TokenTracker,
)

logger = logging.getLogger(__name__)


# ─── Confidence Heuristics ──────────────────────────────────────────────────

_LOW_CONFIDENCE_PATTERNS = [
    re.compile(r"\bI('m| am) not sure\b", re.IGNORECASE),
    re.compile(r"\bI don'?t know\b", re.IGNORECASE),
    re.compile(r"\bI('m| am) unable to\b", re.IGNORECASE),
    re.compile(r"\bcannot determine\b", re.IGNORECASE),
    re.compile(r"\bnot certain\b", re.IGNORECASE),
    re.compile(r"\bI think\b.*\bbut\b", re.IGNORECASE),
    re.compile(r"\bthis is (just )?a guess\b", re.IGNORECASE),
    re.compile(r"\bapproximately\b", re.IGNORECASE),
    re.compile(r"\bI apologize\b", re.IGNORECASE),
    re.compile(r"\bas an AI\b", re.IGNORECASE),
]


def _estimate_confidence(result: CompletionResult, profile: TaskProfile) -> float:
    """
    Heuristic confidence score [0.0 – 1.0] for a completion result.
    Used to decide whether to escalate to a stronger model.
    """
    if not result.success or not result.text.strip():
        return 0.0

    text = result.text
    score = 1.0

    # Penalize low-confidence language
    for pattern in _LOW_CONFIDENCE_PATTERNS:
        if pattern.search(text):
            score -= 0.15

    # Penalize very short responses for non-trivial tasks
    word_count = len(text.split())
    if profile.difficulty in (Difficulty.MEDIUM, Difficulty.HARD):
        if word_count < 20:
            score -= 0.3
        elif word_count < 50:
            score -= 0.1

    # Penalize empty-looking code responses for coding tasks
    if profile.task_type == TaskType.CODING:
        if "```" not in text and "def " not in text and "function" not in text:
            if word_count < 30:
                score -= 0.2

    return max(0.0, min(1.0, score))


# ─── Routing Decision ───────────────────────────────────────────────────────

@dataclass
class RoutingDecision:
    """Captures why a particular model was chosen."""
    model: ModelSpec
    reason: str
    attempt: int = 1


class Router:
    """
    The core routing engine. Routes tasks to the cheapest sufficient model.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.ollama = OllamaProvider(config)
        self.fireworks = FireworksProvider(config)
        self.tracker = TokenTracker()

        # Detect available local models at startup
        self._available_local: list[ModelSpec] = []
        self._init_local_models()

    def _init_local_models(self):
        """Discover which Ollama models are actually installed."""
        if not self.ollama.is_available():
            logger.warning("Ollama is not running — local routing disabled")
            return

        for model in get_local_models():
            if self.ollama.has_model(model.id):
                self._available_local.append(model)
                logger.info(f"✓ Local model available: {model.name}")

        if not self._available_local:
            logger.warning("No local Ollama models found — will use Fireworks only")

    def _pick_local_model(self, profile: TaskProfile) -> Optional[ModelSpec]:
        """Pick the best available local model for this task."""
        if not self._available_local:
            return None

        # For trivial/easy tasks, smallest local model is fine
        if profile.difficulty in (Difficulty.TRIVIAL, Difficulty.EASY):
            return self._available_local[0]

        # For medium/hard, prefer the largest available local model
        # (still zero Fireworks tokens, so always cheaper than any API call)
        return self._available_local[-1] if self._available_local else None

    def _pick_fireworks_model(self, profile: TaskProfile) -> Optional[ModelSpec]:
        """Pick the cheapest sufficient Fireworks model for this task."""
        if profile.difficulty == Difficulty.TRIVIAL:
            return get_cheapest_fireworks(min_tier=Tier.TINY)
        elif profile.difficulty == Difficulty.EASY:
            return get_cheapest_fireworks(min_tier=Tier.TINY)
        elif profile.difficulty == Difficulty.MEDIUM:
            return get_cheapest_fireworks(min_tier=Tier.SMALL)
        else:  # HARD
            return get_cheapest_fireworks(min_tier=Tier.MEDIUM)

    def _get_next_fireworks_model(self, current: ModelSpec) -> Optional[ModelSpec]:
        """Get the next more expensive Fireworks model for escalation."""
        models = sorted(FIREWORKS_MODELS, key=lambda m: (m.cost_score, m.tier))
        found = False
        for m in models:
            if found and m.tier > current.tier:
                return m
            if m.id == current.id:
                found = True
        return None

    def route(
        self,
        task_text: str,
        profile: TaskProfile,
        system_prompt: Optional[str] = None,
        history: Optional[list[dict]] = None,
    ) -> CompletionResult:
        """
        Route a task to the optimal model and return the result.
        Implements the full escalation chain: local → cheap → expensive.
        """
        messages = []
        best_local_result = None
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        # Inject conversation history so the model knows the prior context
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": task_text})

        # ── Step 1: Try local model first (zero Fireworks tokens) ──────
        if self.config.router.prefer_local and self._available_local:
            if self.config.router.enable_committee and profile.difficulty in (Difficulty.MEDIUM, Difficulty.HARD) and len(self._available_local) >= 2:
                committee_size = min(self.config.router.committee_size, len(self._available_local))
                committee = self._available_local[-committee_size:]
                logger.info(f"🔀 Committee Activated: Gathering {len(committee)} drafts for {profile.difficulty.name} task")
                
                drafts = []
                for member in committee:
                    res = self.ollama.complete(
                        model=member,
                        messages=messages,
                        max_tokens=self.config.router.max_completion_tokens,
                        temperature=self.config.router.temperature,
                    )
                    if res.success:
                        drafts.append(f"--- Draft from {member.name} ---\n{res.text}")
                        self.tracker.record(res)
                        logger.info(f"  ✓ Draft gathered from {member.name}")
                    else:
                        logger.warning(f"  ✗ Draft failed from {member.name}: {res.error}")

                if drafts:
                    synth_model_spec = next((m for m in self._available_local if m.id == self.config.router.synthesizer_model), self._available_local[-1])
                    
                    synth_prompt = (
                        "You are a master synthesizer. The user asked a question, and here are several draft answers from other AI models.\n"
                        "Analyze them, identify the correct logic, and produce a single perfect final answer. Do NOT mention that you are a synthesizer or refer to the drafts in your final answer.\n\n"
                        + "\n\n".join(drafts)
                    )
                    
                    synth_messages = messages.copy()
                    synth_messages.append({"role": "system", "content": synth_prompt})
                    
                    logger.info(f"🔀 Synthesizer running: {synth_model_spec.name}")
                    result = self.ollama.complete(
                        model=synth_model_spec,
                        messages=synth_messages,
                        max_tokens=self.config.router.max_completion_tokens,
                        temperature=self.config.router.temperature,
                    )
                    
                    if result.success:
                        confidence = _estimate_confidence(result, profile)
                        logger.info(
                            f"  ✓ Committee result: {len(result.text)} chars, "
                            f"confidence={confidence:.2f}, "
                            f"latency={result.latency_ms:.0f}ms"
                        )
                        self.tracker.record(result)
                        if confidence >= self.config.router.confidence_threshold or not self.config.router.enable_escalation:
                            return result
                        else:
                            logger.info(f"  ⚠ Low confidence ({confidence:.2f}) — escalating to Fireworks")
                            best_local_result = result
                    else:
                        logger.warning(f"  ✗ Synthesizer failed: {result.error}")
                else:
                    logger.warning("  ✗ All committee members failed to generate drafts")
            else:
                local_model = self._pick_local_model(profile)
                if local_model:
                    decision = RoutingDecision(
                        model=local_model,
                        reason=f"Local-first: {profile.difficulty.name} task → {local_model.name}",
                    )
                    logger.info(f"🔀 Routing to LOCAL: {decision.reason}")

                    result = self.ollama.complete(
                        model=local_model,
                        messages=messages,
                        max_tokens=self.config.router.max_completion_tokens,
                        temperature=self.config.router.temperature,
                    )

                    if result.success:
                        confidence = _estimate_confidence(result, profile)
                        logger.info(
                            f"  ✓ Local result: {len(result.text)} chars, "
                            f"confidence={confidence:.2f}, "
                            f"latency={result.latency_ms:.0f}ms"
                        )

                        self.tracker.record(result)
                        if confidence >= self.config.router.confidence_threshold or not self.config.router.enable_escalation:
                            return result
                        else:
                            logger.info(
                                f"  ⚠ Low confidence ({confidence:.2f}) — escalating to Fireworks"
                            )
                            best_local_result = result
                    else:
                        logger.warning(f"  ✗ Local model failed: {result.error}")

        # ── Step 2: Try cheapest sufficient Fireworks model ────────────
        if not self.fireworks.is_available:
            if best_local_result:
                logger.warning("Fireworks API not available. Falling back to low-confidence local result.")
                return best_local_result
                
            logger.error("Fireworks API not available and local model failed")
            return CompletionResult(
                text="Error: No models available",
                model_id="none",
                provider="none",
                success=False,
                error="Neither local nor Fireworks models are available",
            )

        fw_model = self._pick_fireworks_model(profile)
        if fw_model:
            decision = RoutingDecision(
                model=fw_model,
                reason=f"Cheapest Fireworks for {profile.difficulty.name}: {fw_model.name}",
            )
            logger.info(f"🔀 Routing to FIREWORKS: {decision.reason}")

            result = self.fireworks.complete(
                model=fw_model,
                messages=messages,
                max_tokens=self.config.router.max_completion_tokens,
                temperature=self.config.router.temperature,
            )

            if result.success:
                confidence = _estimate_confidence(result, profile)
                logger.info(
                    f"  ✓ Fireworks result: {len(result.text)} chars, "
                    f"confidence={confidence:.2f}, "
                    f"tokens={result.total_tokens}, "
                    f"latency={result.latency_ms:.0f}ms"
                )

                if confidence >= self.config.router.confidence_threshold:
                    self.tracker.record(result)
                    return result
                elif self.config.router.enable_escalation:
                    logger.info(f"  ⚠ Low confidence — escalating to larger model")
                else:
                    self.tracker.record(result)
                    return result
            else:
                logger.warning(f"  ✗ Fireworks model failed: {result.error}")

            # ── Step 3: Escalate to a stronger Fireworks model ─────────
            if self.config.router.enable_escalation:
                next_model = self._get_next_fireworks_model(fw_model)
                if next_model:
                    decision = RoutingDecision(
                        model=next_model,
                        reason=f"Escalation from {fw_model.name} → {next_model.name}",
                        attempt=2,
                    )
                    logger.info(f"🔀 ESCALATING: {decision.reason}")

                    result = self.fireworks.complete(
                        model=next_model,
                        messages=messages,
                        max_tokens=self.config.router.max_completion_tokens,
                        temperature=self.config.router.temperature,
                    )
                    self.tracker.record(result)
                    return result

        # ── Fallback: just return whatever we have ─────────────────────
        logger.error("All routing attempts exhausted")
        return CompletionResult(
            text="Error: All models failed",
            model_id="none",
            provider="none",
            success=False,
            error="All routing attempts failed",
        )

    def get_stats(self) -> str:
        """Return a formatted summary of token usage."""
        return self.tracker.summary()
