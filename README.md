# ⚡ Token-Efficient Routing Agent

An AI agent that completes tasks using the **fewest Fireworks AI tokens possible** by intelligently routing between free local models (Ollama) and paid API models (Fireworks AI).

## Architecture

```
Task → Classifier → Router → Model → Response
         (0 tokens)    │
                        ├── Ollama (local, FREE)
                        ├── Fireworks Tiny  ($)
                        ├── Fireworks Small ($$)
                        ├── Fireworks Medium ($$$)
                        └── Fireworks Large ($$$$)
```

### How Routing Works

1. **Classify** — Zero-token heuristics analyze task difficulty (trivial → hard) and type (coding, factual, creative, etc.)
2. **Route Local First** — Always try the local Ollama model first. Zero Fireworks tokens = best score.
3. **Confidence Check** — If the local response looks uncertain, escalate to a cheap Fireworks model.
4. **Escalate if Needed** — If the cheap model also produces low-confidence output, step up to a larger model.

### Key Strategy

> **Local models cost 0 Fireworks tokens.** The agent tries Ollama first for everything. Only tasks that genuinely need API models get routed there — and always to the cheapest one that works.

## Setup

### Prerequisites
- **Python 3.10+**
- **Ollama** installed and running (`ollama serve`)
- **Fireworks AI API key** (for tasks that need it)

### Install

```bash
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env with your Fireworks API key
```

Or set environment variables directly:
```bash
export FIREWORKS_API_KEY="your_key_here"
export OLLAMA_MODEL="llama3.2"       # or any local model you have
```

### Pull an Ollama Model

```bash
ollama pull llama3.2
```

## Usage

### Interactive Mode
```bash
python main.py
```

### Batch Mode (process tasks from file)
```bash
python main.py --tasks example_tasks.json
python main.py --tasks example_tasks.json --output results.json
```

### Evaluate Results
```bash
python main.py --evaluate results.json --references answers.json
```

### Options
```
--tasks, -t          Path to tasks file (JSON or JSONL)
--output, -o         Path to save results
--evaluate, -e       Evaluate results against references
--references, -r     Reference answers for evaluation
--verbose, -v        Verbose logging
--no-local           Disable local models (Fireworks only)
--no-escalation      Disable confidence-based escalation
--confidence-threshold  Set escalation threshold (0.0-1.0)
```

## Task File Format

```json
[
  {"id": "task_1", "prompt": "What is 2+2?"},
  {"id": "task_2", "prompt": "Write a Python sort function"}
]
```

Also supports JSONL (one JSON object per line).

## Project Structure

```
token-router-agent/
├── main.py          # CLI entry point
├── agent.py         # Task orchestrator
├── router.py        # Routing engine with escalation
├── classifier.py    # Zero-token task classifier
├── providers.py     # Ollama & Fireworks API clients
├── models.py        # Model registry with cost tiers
├── evaluator.py     # Local accuracy checker
├── config.py        # Configuration management
├── example_tasks.json
├── requirements.txt
└── .env.example
```

## Scoring

The competition scores on two axes:
- **Token count** — Total Fireworks AI tokens used (lower is better)
- **Accuracy** — Correctness of answers (higher is better)

This agent optimizes for minimal token spend by:
1. Handling everything possible locally (0 tokens)
2. Using the cheapest Fireworks model when API is needed
3. Only escalating to expensive models as a last resort
