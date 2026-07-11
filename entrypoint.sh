#!/bin/bash
set -e

INPUT_PATH=${TASKS_JSON_PATH:-"/input/tasks.json"}

if [ -f "$INPUT_PATH" ]; then
    echo "=========================================="
    echo "  Evaluation Mode: tasks.json detected"
    echo "=========================================="
    exec python evaluate.py
else
    echo "=========================================="
    echo "  Web Mode: Starting Flask Server"
    echo "=========================================="
    exec python app.py
fi
