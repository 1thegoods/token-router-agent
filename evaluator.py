"""
Local Accuracy Evaluator — checks answer quality before submission.

Compares agent outputs against reference answers using multiple
scoring strategies (exact match, fuzzy match, semantic similarity).
"""

import re
import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional


@dataclass
class EvalScore:
    """Score for a single task evaluation."""
    task_id: str
    exact_match: bool
    fuzzy_score: float        # 0.0 – 1.0
    contains_answer: bool     # Reference answer appears in response
    overall_score: float      # Weighted composite score
    details: str


@dataclass
class EvalReport:
    """Aggregate evaluation results."""
    total: int
    scores: list[EvalScore]

    @property
    def avg_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.overall_score for s in self.scores) / len(self.scores)

    @property
    def exact_matches(self) -> int:
        return sum(1 for s in self.scores if s.exact_match)

    @property
    def accuracy(self) -> float:
        """Percentage of tasks with overall_score >= 0.5."""
        if not self.scores:
            return 0.0
        passing = sum(1 for s in self.scores if s.overall_score >= 0.5)
        return passing / len(self.scores)

    def summary(self) -> str:
        return (
            f"╔══ Evaluation Summary ══════════════════════════╗\n"
            f"║  Tasks evaluated:    {self.total:>6}                   ║\n"
            f"║  Exact matches:      {self.exact_matches:>6}                   ║\n"
            f"║  Avg score:          {self.avg_score:>6.3f}                   ║\n"
            f"║  Accuracy (≥0.5):    {self.accuracy:>6.1%}                   ║\n"
            f"╚════════════════════════════════════════════════╝"
        )


def _normalize(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def _fuzzy_match(a: str, b: str) -> float:
    """Fuzzy string similarity using SequenceMatcher."""
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def _extract_core_answer(text: str) -> str:
    """
    Try to extract the 'core answer' from a verbose LLM response.
    Looks for patterns like "The answer is X" or takes the first sentence.
    """
    # Try common answer patterns
    patterns = [
        r"(?:the answer is|answer:)\s*(.+?)(?:\.|$)",
        r"(?:result is|result:)\s*(.+?)(?:\.|$)",
        r"^(.+?)(?:\.|$)",  # First sentence
    ]
    for pat in patterns:
        match = re.search(pat, text.strip(), re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return text.strip()


def evaluate_single(
    task_id: str,
    agent_answer: str,
    reference_answer: str,
) -> EvalScore:
    """Evaluate a single agent answer against a reference."""
    norm_agent = _normalize(agent_answer)
    norm_ref = _normalize(reference_answer)

    # Exact match (after normalization)
    exact = norm_agent == norm_ref

    # Fuzzy similarity
    fuzzy = _fuzzy_match(agent_answer, reference_answer)

    # Check if the reference answer is contained in the response
    contains = norm_ref in norm_agent

    # Also check core extracted answer
    core = _normalize(_extract_core_answer(agent_answer))
    core_match = _fuzzy_match(core, reference_answer)

    # Composite score (weighted)
    overall = max(
        1.0 if exact else 0.0,
        1.0 if contains else 0.0,
        fuzzy,
        core_match,
    )

    details = (
        f"exact={exact}, fuzzy={fuzzy:.3f}, "
        f"contains={contains}, core_match={core_match:.3f}"
    )

    return EvalScore(
        task_id=task_id,
        exact_match=exact,
        fuzzy_score=fuzzy,
        contains_answer=contains,
        overall_score=overall,
        details=details,
    )


def evaluate_batch(
    results: list[dict],
    references: list[dict],
) -> EvalReport:
    """
    Evaluate a batch of agent results against reference answers.

    Args:
        results: List of {"task_id": ..., "answer": ...}
        references: List of {"task_id": ..., "answer": ...}
    """
    ref_map = {r["task_id"]: r["answer"] for r in references if "answer" in r}

    scores = []
    for result in results:
        task_id = result.get("task_id", result.get("id", "unknown"))
        agent_answer = result.get("answer", "")

        if task_id in ref_map:
            score = evaluate_single(task_id, agent_answer, ref_map[task_id])
            scores.append(score)

    return EvalReport(total=len(scores), scores=scores)


def evaluate_from_files(
    results_path: str,
    references_path: str,
) -> EvalReport:
    """Load results and references from JSON files and evaluate."""
    with open(results_path) as f:
        results_data = json.loads(f.read())

    with open(references_path) as f:
        references = json.loads(f.read())

    # Handle nested results format from agent output
    if isinstance(results_data, dict) and "results" in results_data:
        results = results_data["results"]
    else:
        results = results_data

    if not isinstance(references, list):
        references = [references]

    return evaluate_batch(results, references)
