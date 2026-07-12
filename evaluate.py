"""
Evaluation script for AMD Hackathon scoring harness.

Hybrid zero-token approach:
  1. Local math solver         → 0 tokens
  2. Prompt cache (duplicates) → 0 tokens
  3. Local Qwen2.5-3B GGUF    → 0 tokens
  4. Fireworks API fallback    → tokens (only if local model fails)

Contract:
  - Reads tasks from /input/tasks.json
  - Writes results to /output/results.json
  - Exits with code 0
"""

import os
import json
import sys
import re
import time
import logging
import requests

# Try to import llama_cpp for local inference
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ─── Configuration from environment ─────────────────────────────────────────

FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY", "")
FIREWORKS_BASE_URL = os.environ.get(
    "FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"
)
ALLOWED_MODELS_RAW = os.environ.get("ALLOWED_MODELS", "")

INPUT_PATH = os.environ.get("TASKS_JSON_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("RESULTS_JSON_PATH", "/output/results.json")

LOCAL_MODEL_PATH = os.environ.get(
    "LOCAL_MODEL_PATH", "/app/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
)

# ─── Model tiers for API fallback (cheapest first) ──────────────────────────

MODEL_TIERS = [
    "accounts/fireworks/models/llama-v3p2-1b-instruct",
    "accounts/fireworks/models/llama-v3p2-3b-instruct",
    "accounts/fireworks/models/llama-v3p1-8b-instruct",
    "accounts/fireworks/models/llama4-scout-instruct-basic",
    "accounts/fireworks/models/qwen2p5-72b-instruct",
    "accounts/fireworks/models/llama-v3p1-70b-instruct",
    "accounts/fireworks/models/deepseek-v3",
    "accounts/fireworks/models/llama-v3p1-405b-instruct",
    "accounts/fireworks/models/llama4-maverick-instruct-basic",
]


def parse_allowed_models() -> list[str]:
    """Parse the ALLOWED_MODELS env var into a list of model IDs."""
    if not ALLOWED_MODELS_RAW:
        return []
    raw = ALLOWED_MODELS_RAW.strip()
    if raw.startswith("["):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return [m.strip() for m in raw.split(",") if m.strip()]


def pick_models(allowed: list[str]) -> dict:
    """Pick a 'cheap' and 'strong' model from the allowed list."""
    if not allowed:
        return {
            "cheap": "accounts/fireworks/models/llama-v3p2-3b-instruct",
            "strong": "accounts/fireworks/models/llama-v3p1-70b-instruct",
        }
    tier_order = {m: i for i, m in enumerate(MODEL_TIERS)}
    sorted_models = sorted(allowed, key=lambda m: tier_order.get(m, 999))
    return {
        "cheap": sorted_models[0],
        "strong": sorted_models[-1] if len(sorted_models) > 1 else sorted_models[0],
    }


# ─── Lightweight task classifier (no LLM needed) ────────────────────────────

SIMPLE_PATTERNS = [
    re.compile(r"\b(hello|hi|hey|greet|thanks|thank you)\b", re.I),
    re.compile(r"\b(what is|define|who is|when was|where is)\b", re.I),
    re.compile(r"\b(sentiment|classify|categorize|label)\b", re.I),
    re.compile(r"\b(named entit|NER|extract names|extract entities)\b", re.I),
    re.compile(r"\b(translate|convert .* to)\b", re.I),
    re.compile(r"\b(true or false|yes or no)\b", re.I),
    re.compile(r"\b(list|enumerate)\b", re.I),
]

HARD_PATTERNS = [
    re.compile(r"\b(write .* code|implement|debug|fix .* bug|refactor)\b", re.I),
    re.compile(r"\b(algorithm|complexity|big-?O|recursive)\b", re.I),
    re.compile(r"\b(prove|theorem|mathematical|calculus|integral)\b", re.I),
    re.compile(r"\b(explain .* detail|step.by.step|reason|logic|deduc)\b", re.I),
    re.compile(r"\b(summar|synthesiz|analyz|compar|contrast|evaluat)\b", re.I),
    re.compile(r"```", re.I),
    re.compile(r"\b(python|javascript|java|rust|cpp|sql|html)\b", re.I),
]


def classify_task(prompt: str) -> str:
    """Classify a task as 'simple' or 'hard' using regex heuristics."""
    word_count = len(prompt.split())
    hard_score = sum(1 for p in HARD_PATTERNS if p.search(prompt))
    simple_score = sum(1 for p in SIMPLE_PATTERNS if p.search(prompt))
    if word_count > 150 or "```" in prompt:
        return "hard"
    if hard_score > simple_score:
        return "hard"
    return "simple"


# ─── Local Solvers (0 Tokens) ───────────────────────────────────────────────

def solve_math_locally(prompt: str) -> str | None:
    """Solve simple binary math operations without an API."""
    prompt_clean = (
        prompt.lower()
        .replace("what is", "")
        .replace("calculate", "")
        .replace("solve", "")
        .replace("?", "")
        .strip()
    )
    match = re.search(r"^([\d\.]+)\s*([\+\-\*\/])\s*([\d\.]+)$", prompt_clean)
    if match:
        num1, op, num2 = match.groups()
        try:
            n1 = float(num1) if '.' in num1 else int(num1)
            n2 = float(num2) if '.' in num2 else int(num2)
            if op == '+': return str(n1 + n2)
            if op == '-': return str(n1 - n2)
            if op == '*': return str(n1 * n2)
            if op == '/': return str(n1 / n2) if n2 != 0 else None
        except Exception:
            return None
    return None


# ─── Local Model (0 Tokens) ─────────────────────────────────────────────────

def init_local_model():
    """Load the local GGUF model. Returns Llama instance or None."""
    if not LLAMA_CPP_AVAILABLE:
        logger.warning("llama-cpp-python not installed — skipping local model")
        return None
    if not os.path.exists(LOCAL_MODEL_PATH):
        logger.warning(f"Model file not found: {LOCAL_MODEL_PATH}")
        return None
    try:
        n_threads = max(1, (os.cpu_count() or 4) // 2)
        logger.info(f"Loading local model: {LOCAL_MODEL_PATH} ({n_threads} threads)")
        llm = Llama(
            model_path=LOCAL_MODEL_PATH,
            n_ctx=2048,
            n_threads=n_threads,
            verbose=False,
        )
        logger.info("Local model loaded successfully!")
        return llm
    except Exception as e:
        logger.error(f"Failed to load local model: {e}")
        return None


def call_local_model(llm, prompt: str, max_tokens: int = 256) -> str | None:
    """Run inference on the local model. Returns answer string or None."""
    try:
        response = llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "Answer directly. Be concise. Provide only what is requested."
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.1,
        )
        answer = response["choices"][0]["message"]["content"]
        if answer and answer.strip():
            return answer.strip()
        return None
    except Exception as e:
        logger.error(f"Local model inference error: {e}")
        return None


# ─── Fireworks API caller (fallback) ────────────────────────────────────────

def call_fireworks(
    model: str,
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.1,
    timeout: int = 60,
) -> dict:
    """Call the Fireworks AI API directly."""
    url = f"{FIREWORKS_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {FIREWORKS_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Answer directly. Be concise. Provide only what is requested."
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    choice = data["choices"][0]
    usage = data.get("usage", {})
    return {
        "answer": choice["message"]["content"],
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "model": model,
    }


# ─── Main evaluation loop ───────────────────────────────────────────────────

def process_task(
    task: dict, models: dict, idx: int, total: int, local_llm=None
) -> dict:
    """Process a single task. Never raises — always returns a result dict."""
    task_id = task.get("task_id", f"task_{idx}")
    prompt = task.get("prompt", "")

    logger.info(f"[{idx}/{total}] Processing task {task_id}")
    start = time.time()

    try:
        # ── Step 1: Try local math solver (0 tokens) ──
        local_answer = solve_math_locally(prompt)
        if local_answer is not None:
            elapsed = time.time() - start
            logger.info(f"  → Solved with math solver! 0 tokens | {elapsed:.1f}s")
            return {"task_id": task_id, "answer": local_answer}

        # ── Step 2: Try local LLM (0 tokens) ──
        if local_llm is not None:
            difficulty = classify_task(prompt)
            max_tokens = 256 if difficulty == "simple" else 512
            logger.info(f"  → Local model (difficulty={difficulty})...")
            answer = call_local_model(local_llm, prompt, max_tokens=max_tokens)
            if answer is not None:
                elapsed = time.time() - start
                logger.info(f"  ✓ Local model done in {elapsed:.1f}s | 0 tokens")
                return {"task_id": task_id, "answer": answer}
            logger.warning("  ⚠ Local model returned empty — falling back to API")

        # ── Step 3: Fireworks API fallback ──
        if not FIREWORKS_API_KEY:
            logger.error("  ✗ No API key and local model failed")
            return {"task_id": task_id, "answer": "Error: no model available"}

        difficulty = classify_task(prompt)
        model = models["cheap"] if difficulty == "simple" else models["strong"]
        max_tokens = 512 if difficulty == "simple" else 1536

        logger.info(f"  → API fallback: {model}")
        result = call_fireworks(model, prompt, max_tokens=max_tokens)

        elapsed = time.time() - start
        logger.info(
            f"  ✓ Done in {elapsed:.1f}s | "
            f"tokens: {result['input_tokens']}in + {result['output_tokens']}out"
        )
        return {"task_id": task_id, "answer": result["answer"]}

    except requests.exceptions.Timeout:
        logger.warning(f"  ✗ Timeout for task {task_id}")
        return {"task_id": task_id, "answer": "Error: request timed out"}

    except requests.exceptions.HTTPError as e:
        logger.warning(f"  ✗ HTTP error for task {task_id}: {e}")
        try:
            fallback = models["cheap"]
            logger.info(f"  ↻ Retrying with {fallback}")
            result = call_fireworks(fallback, prompt, max_tokens=512)
            return {"task_id": task_id, "answer": result["answer"]}
        except Exception as e2:
            logger.error(f"  ✗ Retry also failed: {e2}")
            return {"task_id": task_id, "answer": f"Error: {e}"}

    except Exception as e:
        logger.error(f"  ✗ Unexpected error for task {task_id}: {e}")
        return {"task_id": task_id, "answer": f"Error: {e}"}


def main():
    logger.info("=" * 60)
    logger.info("  AMD Hackathon — Hybrid Zero-Token Router")
    logger.info("=" * 60)

    # ── Initialize local model ──
    local_llm = init_local_model()
    if local_llm:
        logger.info("✓ Local model ready — targeting 0 API tokens!")
    else:
        logger.info("⚠ No local model — will use Fireworks API")

    # ── Check API fallback ──
    if not FIREWORKS_API_KEY:
        if local_llm is None:
            logger.error("No local model AND no API key!")
            os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
            with open(OUTPUT_PATH, "w") as f:
                json.dump([], f)
            sys.exit(0)
        logger.info("No API key — running in pure local mode")
    else:
        logger.info(f"API fallback available: {FIREWORKS_BASE_URL}")

    # ── Parse allowed models for API fallback ──
    allowed = parse_allowed_models()
    models = pick_models(allowed)
    logger.info(f"API cheap:  {models['cheap']}")
    logger.info(f"API strong: {models['strong']}")

    # ── Read tasks ──
    if not os.path.exists(INPUT_PATH):
        logger.error(f"Input file not found: {INPUT_PATH}")
        os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            json.dump([], f)
        sys.exit(0)

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    if not isinstance(tasks, list):
        logger.error("tasks.json is not a list")
        os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            json.dump([], f)
        sys.exit(0)

    logger.info(f"Loaded {len(tasks)} tasks")

    # ── Process tasks ──
    results = []
    cache = {}
    for idx, task in enumerate(tasks, 1):
        prompt = task.get("prompt", "")
        task_id = task.get("task_id", f"task_{idx}")

        # Check cache for exact duplicate prompts (0 tokens)
        if prompt in cache:
            logger.info(f"[{idx}/{len(tasks)}] Cache hit for task {task_id}")
            results.append({"task_id": task_id, "answer": cache[prompt]})
            continue

        result = process_task(task, models, idx, len(tasks), local_llm=local_llm)
        cache[prompt] = result["answer"]
        results.append(result)

    # ── Write output ──
    out_dir = os.path.dirname(OUTPUT_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Wrote {len(results)} results to {OUTPUT_PATH}")
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
