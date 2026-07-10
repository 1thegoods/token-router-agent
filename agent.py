"""
Agent — the orchestrator that processes tasks end-to-end.

Loads tasks (from file or stdin), classifies them, routes them through
the Router, and collects results with full token accounting.
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from config import AgentConfig, load_config
from classifier import classify, TaskProfile
from router import Router
from providers import CompletionResult

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Result for a single processed task."""
    task_id: str
    task_text: str
    profile: dict
    answer: str
    model_used: str
    provider: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    success: bool


@dataclass
class AgentReport:
    """Final report after processing all tasks."""
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    results: list[TaskResult] = field(default_factory=list)
    total_time_ms: float = 0.0
    token_summary: str = ""


class Agent:
    """
    Token-efficient routing agent.

    Usage:
        agent = Agent()
        report = agent.run_tasks(tasks)
        print(report.token_summary)
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or load_config()
        self.router = Router(self.config)

        # Default system prompt — kept minimal to save tokens
        self.system_prompt = (
            "Answer accurately and concisely. "
            "Provide only the essential information requested. "
            "Do not add unnecessary disclaimers or preambles."
        )

    def process_task(self, task_id: str, task_text: str, history: Optional[list[dict]] = None) -> TaskResult:
        """Process a single task through classify → route → complete."""

        # Step 1: Classify (zero tokens)
        profile = classify(task_text)
        if self.config.verbose:
            logger.info(f"📋 Task [{task_id}]: {profile.reasoning}")

        # Step 2: Route and complete
        result = self.router.route(
            task_text=task_text,
            profile=profile,
            system_prompt=self.system_prompt,
            history=history,
        )

        # Step 3: Package result
        return TaskResult(
            task_id=task_id,
            task_text=task_text[:200],  # Truncate for logging
            profile={
                "difficulty": profile.difficulty.name,
                "task_type": profile.task_type.name,
                "estimated_output_tokens": profile.estimated_output_tokens,
            },
            answer=result.text,
            model_used=result.model_id,
            provider=result.provider,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=result.latency_ms,
            success=result.success,
        )

    def run_tasks(self, tasks: list[dict]) -> AgentReport:
        """
        Process a batch of tasks.

        Args:
            tasks: List of dicts with 'id' and 'prompt' keys.
                   Example: [{"id": "task_1", "prompt": "What is 2+2?"}]

        Returns:
            AgentReport with all results and token summary.
        """
        report = AgentReport()
        start_time = time.perf_counter()

        logger.info(f"🚀 Starting agent with {len(tasks)} tasks")
        logger.info(f"   Local models: {len(self.router._available_local)}")
        logger.info(f"   Fireworks available: {self.router.fireworks.is_available}")
        logger.info(f"   Escalation: {'enabled' if self.config.router.enable_escalation else 'disabled'}")
        print()

        for i, task in enumerate(tasks, 1):
            task_id = task.get("id", f"task_{i}")
            task_text = task.get("prompt", task.get("question", task.get("text", "")))

            if not task_text:
                logger.warning(f"Skipping task {task_id}: no prompt found")
                continue

            print(f"── Task {i}/{len(tasks)}: {task_id} {'─' * 40}")
            result = self.process_task(task_id, task_text)
            report.results.append(result)

            if result.success:
                report.successful_tasks += 1
                preview = result.answer[:120].replace("\n", " ")
                print(f"   ✓ [{result.provider}:{result.model_used.split('/')[-1]}] "
                      f"tokens={result.input_tokens}+{result.output_tokens} "
                      f"latency={result.latency_ms:.0f}ms")
                print(f"   → {preview}...")
            else:
                report.failed_tasks += 1
                print(f"   ✗ FAILED: {result.model_used}")
            print()

        report.total_tasks = len(tasks)
        report.total_time_ms = (time.perf_counter() - start_time) * 1000
        report.token_summary = self.router.get_stats()

        return report

    def run_interactive(self):
        """Run in interactive mode — type tasks one at a time. Maintains conversation history."""
        print("╔══════════════════════════════════════════════════╗")
        print("║    Token-Efficient Routing Agent (Interactive)   ║")
        print("║    Type a task, press Enter. Type 'quit' to exit ║")
        print("║    🧠 Conversation memory is ON                   ║")
        print("╚══════════════════════════════════════════════════╝")
        print()

        task_num = 0
        # Conversation history — keeps the last 10 turns to avoid overflowing context
        MAX_HISTORY_TURNS = 10
        history: list[dict] = []

        while True:
            try:
                task_text = input("You > ").strip()
                if not task_text:
                    continue
                if task_text.lower() in ("quit", "exit", "q"):
                    break
                if task_text.lower() in ("clear", "reset"):
                    history.clear()
                    print("\n🗑️  Conversation cleared. Starting fresh!\n")
                    continue

                task_num += 1
                result = self.process_task(f"interactive_{task_num}", task_text, history=history)

                print(f"\n[{result.provider}:{result.model_used.split('/')[-1]}] "
                      f"(tokens: {result.input_tokens}+{result.output_tokens})")
                print(f"\n{result.answer}\n")

                # Append this turn to history so future messages have context
                if result.success and result.answer:
                    history.append({"role": "user", "content": task_text})
                    history.append({"role": "assistant", "content": result.answer})
                    # Trim to last N turns (each turn = 2 messages)
                    if len(history) > MAX_HISTORY_TURNS * 2:
                        history = history[-(MAX_HISTORY_TURNS * 2):]

            except (KeyboardInterrupt, EOFError):
                break

        print(f"\n{self.router.get_stats()}")


def load_tasks_from_file(filepath: str) -> list[dict]:
    """Load tasks from a JSON or JSONL file."""
    path = Path(filepath)

    if path.suffix == ".jsonl":
        tasks = []
        with open(path) as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    task = json.loads(line)
                    if "id" not in task:
                        task["id"] = f"task_{i}"
                    tasks.append(task)
        return tasks
    else:
        with open(path) as f:
            data = json.loads(f.read())
        if isinstance(data, list):
            for i, task in enumerate(data, 1):
                if "id" not in task:
                    task["id"] = f"task_{i}"
            return data
        else:
            return [data]


def save_results(report: AgentReport, output_path: str):
    """Save agent results to a JSON file."""
    output = {
        "summary": {
            "total_tasks": report.total_tasks,
            "successful": report.successful_tasks,
            "failed": report.failed_tasks,
            "total_time_ms": report.total_time_ms,
        },
        "results": [
            {
                "task_id": r.task_id,
                "answer": r.answer,
                "model_used": r.model_used,
                "provider": r.provider,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "latency_ms": r.latency_ms,
                "success": r.success,
                "classification": r.profile,
            }
            for r in report.results
        ],
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Results saved to {output_path}")
