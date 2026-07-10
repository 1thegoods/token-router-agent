"""
Task Classifier — zero-token, rule-based complexity analysis.

Classifies each incoming task by difficulty level and type so the router
can pick the cheapest model capable of handling it. This classifier uses
NO LLM calls — it's pure heuristics, costing zero tokens.
"""

import re
from dataclasses import dataclass
from enum import Enum, auto


class Difficulty(Enum):
    """How hard is this task for an LLM?"""
    TRIVIAL = auto()   # Greetings, simple lookups, yes/no
    EASY = auto()      # Basic Q&A, short factual answers, simple math
    MEDIUM = auto()    # Multi-step reasoning, summarization, moderate code
    HARD = auto()      # Complex reasoning, long generation, advanced code


class TaskType(Enum):
    """What kind of task is this?"""
    FACTUAL = auto()       # Fact-based Q&A
    REASONING = auto()     # Logic, math, multi-step thinking
    CODING = auto()        # Code generation, debugging, explanation
    CREATIVE = auto()      # Creative writing, brainstorming
    SUMMARIZATION = auto() # Summarize text
    EXTRACTION = auto()    # Extract structured data from text
    TRANSLATION = auto()   # Language translation
    CONVERSATION = auto()  # Casual chat, greetings
    CLASSIFICATION = auto()# Categorize or label items
    UNKNOWN = auto()


@dataclass
class TaskProfile:
    """Analysis result for a single task."""
    difficulty: Difficulty
    task_type: TaskType
    estimated_output_tokens: int
    reasoning: str  # Brief explanation of classification


# ─── Keyword / Pattern Banks ────────────────────────────────────────────────

_GREETING_PATTERNS = re.compile(
    r"^(hi|hello|hey|greetings|howdy|good\s+(morning|afternoon|evening)|what'?s up)\b",
    re.IGNORECASE,
)

_CODING_KEYWORDS = {
    "code", "function", "class", "implement", "debug", "fix", "bug",
    "python", "javascript", "typescript", "java", "rust", "golang", "go",
    "c++", "sql", "html", "css", "react", "api", "endpoint", "algorithm",
    "regex", "parse", "compile", "script", "refactor", "optimize",
    "write a program", "write code", "coding", "programming",
}

_REASONING_KEYWORDS = {
    "why", "explain", "analyze", "compare", "contrast", "evaluate",
    "reason", "logic", "proof", "prove", "derive", "calculate",
    "step by step", "think through", "solve", "deduce", "infer",
    "what if", "would", "could", "should", "trade-off", "pros and cons",
}

_CREATIVE_KEYWORDS = {
    "write a story", "poem", "creative", "imagine", "fiction",
    "brainstorm", "ideas for", "design", "invent", "compose",
    "narrative", "dialogue", "screenplay", "lyrics",
}

_SUMMARIZATION_KEYWORDS = {
    "summarize", "summary", "tldr", "tl;dr", "brief", "overview",
    "key points", "main ideas", "condense", "shorten",
}

_EXTRACTION_KEYWORDS = {
    "extract", "parse", "pull out", "identify", "list all",
    "find all", "structured", "json", "table", "csv",
}

_TRANSLATION_KEYWORDS = {
    "translate", "translation", "in spanish", "in french", "in german",
    "in japanese", "in chinese", "in arabic", "to english", "from english",
}

_COMPLEXITY_BOOSTERS = {
    "detailed", "comprehensive", "thorough", "in-depth", "elaborate",
    "complex", "advanced", "sophisticated", "production-ready",
    "enterprise", "scalable", "complete", "full",
}

_SIMPLICITY_SIGNALS = {
    "simple", "quick", "short", "brief", "one-liner", "easy",
    "basic", "trivial", "just", "only",
}


def _count_keyword_hits(text: str, keywords: set[str]) -> int:
    """Count how many keywords appear in the text (whole-word matching)."""
    text_lower = text.lower()
    count = 0
    for kw in keywords:
        # Multi-word keywords use simple substring match
        if " " in kw:
            if kw in text_lower:
                count += 1
        else:
            # Single-word keywords use word-boundary regex
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                count += 1
    return count


def _detect_task_type(text: str) -> TaskType:
    """Determine the primary task type from the query text."""
    if _GREETING_PATTERNS.match(text.strip()):
        return TaskType.CONVERSATION

    scores = {
        TaskType.CODING: _count_keyword_hits(text, _CODING_KEYWORDS),
        TaskType.REASONING: _count_keyword_hits(text, _REASONING_KEYWORDS),
        TaskType.CREATIVE: _count_keyword_hits(text, _CREATIVE_KEYWORDS),
        TaskType.SUMMARIZATION: _count_keyword_hits(text, _SUMMARIZATION_KEYWORDS),
        TaskType.EXTRACTION: _count_keyword_hits(text, _EXTRACTION_KEYWORDS),
        TaskType.TRANSLATION: _count_keyword_hits(text, _TRANSLATION_KEYWORDS),
    }

    # Check for code blocks in the prompt (strong coding signal)
    if "```" in text or re.search(r"def |function |class |import |const |let |var ", text):
        scores[TaskType.CODING] += 3

    best_type = max(scores, key=scores.get)
    if scores[best_type] == 0:
        # No strong signal — check if it's a question
        if text.strip().endswith("?") or text.lower().startswith(("what", "who", "when", "where", "how")):
            return TaskType.FACTUAL
        return TaskType.UNKNOWN

    return best_type


def _estimate_difficulty(text: str, task_type: TaskType) -> Difficulty:
    """Estimate task difficulty from heuristic signals."""
    # Trivial: greetings and very short queries
    if task_type == TaskType.CONVERSATION:
        return Difficulty.TRIVIAL
    if len(text.split()) < 5 and task_type == TaskType.FACTUAL:
        return Difficulty.TRIVIAL

    word_count = len(text.split())
    complexity_hits = _count_keyword_hits(text, _COMPLEXITY_BOOSTERS)
    simplicity_hits = _count_keyword_hits(text, _SIMPLICITY_SIGNALS)

    # Start with a base score
    score = 0

    # Length-based scoring
    if word_count > 200:
        score += 2  # Long prompts usually need stronger models
    elif word_count > 80:
        score += 1

    # Complexity keywords
    score += complexity_hits
    score -= simplicity_hits

    # Task-type adjustments
    if task_type == TaskType.CODING:
        score += 1  # Code gen generally needs more capability
    if task_type == TaskType.CREATIVE:
        score += 1  # Creative tasks benefit from larger models
    if task_type == TaskType.CLASSIFICATION:
        score -= 1  # Classification is usually straightforward
    if task_type == TaskType.EXTRACTION:
        score -= 1  # Extraction is pattern-based

    # Multi-part question detection
    question_marks = text.count("?")
    numbered_items = len(re.findall(r"^\s*\d+[\.\)]\s", text, re.MULTILINE))
    if question_marks > 2 or numbered_items > 3:
        score += 1

    # Map score to difficulty
    if score <= 0:
        return Difficulty.EASY
    elif score <= 2:
        return Difficulty.MEDIUM
    else:
        return Difficulty.HARD


def _estimate_output_tokens(text: str, task_type: TaskType, difficulty: Difficulty) -> int:
    """Rough estimate of how many output tokens the response will need."""
    base = {
        Difficulty.TRIVIAL: 30,
        Difficulty.EASY: 100,
        Difficulty.MEDIUM: 300,
        Difficulty.HARD: 800,
    }[difficulty]

    multiplier = {
        TaskType.CODING: 1.5,
        TaskType.CREATIVE: 1.8,
        TaskType.SUMMARIZATION: 0.6,
        TaskType.EXTRACTION: 0.8,
        TaskType.CLASSIFICATION: 0.3,
        TaskType.CONVERSATION: 0.4,
        TaskType.TRANSLATION: 1.0,
        TaskType.FACTUAL: 0.8,
        TaskType.REASONING: 1.3,
        TaskType.UNKNOWN: 1.0,
    }[task_type]

    return int(base * multiplier)


def classify(task_text: str) -> TaskProfile:
    """
    Classify a task's difficulty and type using zero-token heuristics.
    Returns a TaskProfile with the analysis results.
    """
    task_type = _detect_task_type(task_text)
    difficulty = _estimate_difficulty(task_text, task_type)
    est_tokens = _estimate_output_tokens(task_text, task_type, difficulty)

    reasoning = (
        f"Type={task_type.name}, Difficulty={difficulty.name}, "
        f"Words={len(task_text.split())}, EstOutputTokens={est_tokens}"
    )

    return TaskProfile(
        difficulty=difficulty,
        task_type=task_type,
        estimated_output_tokens=est_tokens,
        reasoning=reasoning,
    )
