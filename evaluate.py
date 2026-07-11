import os
import json
import sys
import logging
import time

from config import load_config
from multi_agent import AgenticSystem

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    # Paths configured by the hackathon evaluation harness
    input_path = os.environ.get("TASKS_JSON_PATH", "/input/tasks.json")
    output_path = os.environ.get("RESULTS_JSON_PATH", "/output/results.json")

    logger.info(f"Checking for tasks at: {input_path}")
    if not os.path.exists(input_path):
        logger.error(f"Input file not found at {input_path}")
        sys.exit(1)

    logger.info("Loading tasks...")
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            tasks = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read tasks.json: {e}")
        sys.exit(1)

    if not isinstance(tasks, list):
        logger.error("Expected tasks.json to be a list of task objects.")
        sys.exit(1)

    logger.info(f"Loaded {len(tasks)} tasks.")

    # Initialize the Agent
    logger.info("Initializing AgenticSystem...")
    config = load_config()
    agent_sys = AgenticSystem(config)

    results = []

    for idx, task in enumerate(tasks, 1):
        task_id = task.get("task_id", f"task_{idx}")
        prompt = task.get("prompt", "")

        logger.info(f"--- Processing Task {idx}/{len(tasks)} (ID: {task_id}) ---")
        start_time = time.time()
        
        try:
            # Route and process task
            completion = agent_sys.run_task(prompt=prompt)
            answer = completion.text
        except Exception as e:
            logger.error(f"Error processing task {task_id}: {e}")
            answer = f"Error: {e}"

        elapsed = time.time() - start_time
        logger.info(f"Finished task {task_id} in {elapsed:.2f}s")

        results.append({
            "task_id": task_id,
            "answer": answer
        })

    # Write output
    logger.info(f"Writing results to {output_path}")
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to write results.json: {e}")
        sys.exit(1)

    logger.info("Evaluation complete!")

if __name__ == "__main__":
    main()
