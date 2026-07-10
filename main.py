#!/usr/bin/env python3
"""
Token-Efficient Routing Agent — CLI Entry Point

Usage:
    # Interactive mode
    python main.py

    # Batch mode — process tasks from file
    python main.py --tasks tasks.json

    # Batch mode with output
    python main.py --tasks tasks.json --output results.json

    # Evaluate results against reference answers
    python main.py --evaluate results.json --references answers.json

    # Verbose mode
    python main.py --tasks tasks.json --verbose
"""

import argparse
import logging
import sys
import io
import json
from pathlib import Path

# Force UTF-8 encoding for Windows terminals to support UI borders
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from config import load_config
from agent import Agent, load_tasks_from_file, save_results
from evaluator import evaluate_from_files


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[logging.StreamHandler()],
    )


def print_banner():
    print(r"""
  ╔══════════════════════════════════════════════════════════════╗
  ║                                                              ║
  ║   ⚡ Token-Efficient Routing Agent                           ║
  ║                                                              ║
  ║   Routes tasks to the cheapest model that can handle them.   ║
  ║   Local models = 0 Fireworks tokens. That's the goal.        ║
  ║                                                              ║
  ╚══════════════════════════════════════════════════════════════╝
    """)


def main():
    parser = argparse.ArgumentParser(
        description="Token-Efficient Routing Agent — minimize Fireworks API tokens",
    )
    parser.add_argument(
        "--tasks", "-t",
        type=str,
        help="Path to tasks file (JSON or JSONL)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--evaluate", "-e",
        type=str,
        help="Path to results file to evaluate",
    )
    parser.add_argument(
        "--references", "-r",
        type=str,
        help="Path to reference answers for evaluation",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--no-local",
        action="store_true",
        help="Disable local models (Fireworks only)",
    )
    parser.add_argument(
        "--no-escalation",
        action="store_true",
        help="Disable confidence-based escalation",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Confidence threshold for escalation (0.0-1.0)",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    # ── Evaluation mode ──────────────────────────────────────────────
    if args.evaluate:
        if not args.references:
            print("Error: --references required with --evaluate")
            sys.exit(1)

        print_banner()
        print("📊 Evaluating results...\n")

        report = evaluate_from_files(args.evaluate, args.references)
        print(report.summary())

        for score in report.scores:
            status = "✓" if score.overall_score >= 0.5 else "✗"
            print(f"  {status} {score.task_id}: {score.overall_score:.3f} ({score.details})")

        sys.exit(0)

    # ── Agent mode ───────────────────────────────────────────────────
    config = load_config()
    config.verbose = args.verbose

    if args.no_local:
        config.router.prefer_local = False
    if args.no_escalation:
        config.router.enable_escalation = False
    if args.confidence_threshold is not None:
        config.router.confidence_threshold = args.confidence_threshold

    print_banner()

    agent = Agent(config)

    if args.tasks:
        # Batch mode
        tasks = load_tasks_from_file(args.tasks)
        print(f"📂 Loaded {len(tasks)} tasks from {args.tasks}\n")

        report = agent.run_tasks(tasks)

        print(f"\n{'═' * 52}")
        print(report.token_summary)
        print(f"\n⏱  Total time: {report.total_time_ms:.0f}ms")
        print(f"✓  Successful: {report.successful_tasks}/{report.total_tasks}")
        if report.failed_tasks:
            print(f"✗  Failed: {report.failed_tasks}/{report.total_tasks}")

        if args.output:
            save_results(report, args.output)
            print(f"\n💾 Results saved to {args.output}")
    else:
        # Interactive mode
        agent.run_interactive()


if __name__ == "__main__":
    main()
