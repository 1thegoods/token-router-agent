import json
import logging
import concurrent.futures
from typing import List, Dict, Any, Optional
from datetime import datetime

from router import Router
from providers import CompletionResult
from tools import execute_tool, TOOLS_DESCRIPTION

logger = logging.getLogger(__name__)


class AgenticSystem:
    def __init__(self, config):
        self.config = config
        self.router = Router(config)
        # Use synthesizer model for planning/merging if available
        self.planner_model = next(
            (m for m in self.router._available_local if m.id == self.config.router.synthesizer_model), 
            self.router._available_local[-1] if self.router._available_local else None
        )

    def run_task(self, prompt: str, history: Optional[List[Dict]] = None) -> CompletionResult:
        """Main entry point: splits, executes in parallel, and merges."""
        logger.info(f"🚀 Starting agentic execution for prompt: {prompt[:50]}...")
        
        # 1. Split task
        subtasks = self._split_task(prompt)
        if not subtasks:
            # Fallback to simple execution if splitting fails or returns 1 task
            return self._execute_subtask(prompt, history=history)
            
        logger.info(f"🧩 Split into {len(subtasks)} subtasks:")
        for i, st in enumerate(subtasks, 1):
            logger.info(f"  {i}. {st}")

        # 2. Execute parallel
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_task = {executor.submit(self._execute_subtask, task, history=history): task for task in subtasks}
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append((task, result))
                except Exception as exc:
                    logger.error(f"Task {task} generated an exception: {exc}")
                    results.append((task, CompletionResult(
                        text=f"Error executing task: {exc}",
                        model_id="error",
                        provider="none",
                        success=False,
                        error=str(exc)
                    )))

        # 3. Merge
        return self._merge_results(prompt, results)

    def _split_task(self, prompt: str) -> List[str]:
        """Ask LLM to decompose the task into independent subtasks."""
        # To reduce credit and power consumption, we disable parallel splitting 
        # for standard prompts unless explicitly requested.
        return [prompt]

        system_msg = (
            "You are a task planner. Break the user's prompt into a JSON list of independent, atomic sub-tasks. "
            "If the prompt is simple and only requires one action, return an empty list or a list with one item. "
            "Format your response EXACTLY as a JSON list of strings, nothing else. e.g. [\"task 1\", \"task 2\"]"
        )
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ]
        
        res = self.router.ollama.complete(
            model=self.planner_model,
            messages=messages,
            max_tokens=self.config.router.max_completion_tokens,
            temperature=0.1
        )
        
        if not res.success:
            logger.warning(f"Failed to split task: {res.error}")
            return [prompt]
            
        try:
            # Extract JSON if there's markdown wrapping
            text = res.text.strip()
            if text.startswith("```json"):
                text = text.replace("```json", "").replace("```", "").strip()
            elif text.startswith("```"):
                text = text.replace("```", "").strip()
                
            tasks = json.loads(text)
            if isinstance(tasks, list) and len(tasks) > 1:
                return tasks
            return [prompt]
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse split JSON: {res.text}")
            return [prompt]

    def _execute_subtask(self, task_prompt: str, history: Optional[List[Dict]] = None) -> CompletionResult:
        """Run a single subtask with a tool-use loop (Thought -> Tool -> Result)."""
        from classifier import classify, Difficulty, TaskType
        profile = classify(task_prompt)
        
        # Bypass tool loop for simple greetings or trivial tasks
        if profile.task_type == TaskType.CONVERSATION or profile.difficulty == Difficulty.TRIVIAL:
            system_msg = "You are a helpful AI assistant running locally. Answer directly and concisely."
            return self.router.route(
                task_text=task_prompt,
                profile=profile,
                system_prompt=system_msg,
                history=history
            )

        # Tool execution requires strict JSON formatting and reasoning.
        # Small models (used for EASY tasks) fail at this and hallucinate.
        # Elevate EASY tasks to MEDIUM so they are routed to larger models.
        if profile.difficulty == Difficulty.EASY:
            profile.difficulty = Difficulty.MEDIUM
            profile.reasoning += " (Elevated to MEDIUM for tool reliability)"

        system_msg = (
            "You are an autonomous AI agent running locally. Your goal is to ACTUALLY perform the tasks the user requests, NOT just tell them how to do it.\n\n"
            f"{TOOLS_DESCRIPTION}\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            f"0. The current date and time is: {datetime.now().isoformat()}\n"
            "1. If the user asks you to search the web, YOU MUST use the `search_web` tool. DO NOT write code to search the web.\n"
            "2. If the user asks you to write a script or create a file, YOU MUST use the `write_file` tool to save it to their machine. DO NOT just output the code block and tell the user to save it.\n"
            "3. If the user asks you to run a script, YOU MUST use the `run_command` tool.\n"
            "4. You must output exactly ONE [TOOL_CALL] block at a time. Stop and wait for the result.\n"
            "5. Once all tasks are complete, provide your final answer to the user.\n"
            "Failure to use tools when requested means you have failed your job as an agent."
        )
        
        messages = []
        if history:
            messages.extend(history)
            
        messages.append({"role": "system", "content": system_msg})
        messages.append({"role": "user", "content": task_prompt})
        
        max_iterations = 5
        final_result = None
        
        for iteration in range(max_iterations):
            if iteration == 0:
                logger.info("  🧠 Initializing Agent Thought Process")
            else:
                logger.info(f"  🔍 Evaluating Tool Result (Step {iteration+1})")
            current_prompt = messages[-1]["content"]
            current_history = messages[:-1]
            
            res = self.router.route(
                task_text=current_prompt,
                profile=profile,
                system_prompt=system_msg,
                history=current_history
            )
            
            final_result = res
            
            if not res.success:
                logger.error(f"Agent completion failed: {res.error}")
                break
                
            # Parse tool calls
            tool_call, raw_args = self._extract_tool_call(res.text)
            
            if tool_call:
                # Add assistant thought to messages
                messages.append({"role": "assistant", "content": res.text})
                
                # Execute tool
                try:
                    args = json.loads(raw_args)
                    tool_res = execute_tool(tool_call, args)
                    tool_output = f"[TOOL_RESULT] {tool_res.output} [/TOOL_RESULT]"
                    if tool_res.error:
                        tool_output += f"\nError: {tool_res.error}"
                except Exception as e:
                    tool_output = f"[TOOL_RESULT] Error executing {tool_call}: {e} [/TOOL_RESULT]"
                
                # Add tool result to messages
                messages.append({"role": "user", "content": tool_output})
            else:
                # No tool call, we are done
                break
                
        return final_result

    def _extract_tool_call(self, text: str):
        """Extract [TOOL_CALL]{...}[/TOOL_CALL]"""
        import re
        match = re.search(r"\[TOOL_CALL\](.*?)\[/TOOL_CALL\]", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                return data.get("tool"), json.dumps(data.get("args", {}))
            except json.JSONDecodeError:
                return None, None
        return None, None

    def _merge_results(self, original_prompt: str, subtask_results: List[tuple]) -> CompletionResult:
        """Merge all parallel results into a final cohesive response."""
        if not self.planner_model or len(subtask_results) <= 1:
            # If only 1 result, just return it
            return subtask_results[0][1] if subtask_results else CompletionResult(text="No results", model_id="", provider="", success=False)
            
        system_msg = (
            "You are a synthesizer. Merge the sub-task results into a final, coherent answer to the user's original prompt. "
            "If the sub-task results contain errors (e.g., 'Error executing task'), YOU MUST inform the user about the failure clearly "
            "instead of trying to answer the original prompt yourself. Do not hallucinate an answer if the agents failed to do the work."
        )
        
        merged_text = "\n\n".join([f"Subtask: {task}\nResult: {res.text}" for task, res in subtask_results])
        user_msg = f"Original Prompt: {original_prompt}\n\nResults from parallel agents:\n{merged_text}"
        
        res = self.router.ollama.complete(
            model=self.planner_model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=self.config.router.max_completion_tokens,
            temperature=0.3
        )
        
        # Merge metrics
        total_in = sum(r.input_tokens for _, r in subtask_results) + res.input_tokens
        total_out = sum(r.output_tokens for _, r in subtask_results) + res.output_tokens
        max_lat = max([r.latency_ms for _, r in subtask_results] + [res.latency_ms])
        
        return CompletionResult(
            text=res.text,
            model_id=res.model_id,
            provider=res.provider,
            success=res.success,
            input_tokens=total_in,
            output_tokens=total_out,
            latency_ms=max_lat
        )
