"""
Tools — system-level capabilities for the agent.

Gives the agent the ability to read/write files, run commands, and browse
the filesystem on your local machine.
"""

import os
import re
import urllib.request
import urllib.parse
import subprocess
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Safety: restrict file operations to these root directories
ALLOWED_ROOTS = [
    Path.home(),  # User's home directory
]


def _is_safe_path(path: str) -> bool:
    """Check if a path is within allowed directories."""
    resolved = Path(path).resolve()
    return any(resolved.is_relative_to(root) for root in ALLOWED_ROOTS)


# ─── Tool Definitions ───────────────────────────────────────────────────────

TOOLS_DESCRIPTION = """You have access to the following tools to interact with the user's computer and the internet:

1. **read_file** — Read the contents of a file.
   Usage: [TOOL_CALL]{"tool": "read_file", "args": {"path": "C:/path/to/file.py"}}[/TOOL_CALL]

2. **write_file** — Create or overwrite a file with content.
   Usage: [TOOL_CALL]{"tool": "write_file", "args": {"path": "C:/path/to/file.py", "content": "print('hello')"}}[/TOOL_CALL]

3. **run_command** — Run a terminal command and get the output.
   Usage: [TOOL_CALL]{"tool": "run_command", "args": {"command": "python script.py"}}[/TOOL_CALL]

4. **list_dir** — List all files and folders in a directory.
   Usage: [TOOL_CALL]{"tool": "list_dir", "args": {"path": "C:/Users/Steven Hanna/project"}}[/TOOL_CALL]

5. **search_files** — Search for files matching a pattern in a directory.
   Usage: [TOOL_CALL]{"tool": "search_files", "args": {"path": "C:/project", "pattern": "*.py"}}[/TOOL_CALL]

6. **search_web** — Search the internet using DuckDuckGo.
   Usage: [TOOL_CALL]{"tool": "search_web", "args": {"query": "latest AI news"}}[/TOOL_CALL]

7. **read_url** — Read the text content of a webpage.
   Usage: [TOOL_CALL]{"tool": "read_url", "args": {"url": "https://example.com"}}[/TOOL_CALL]

RULES:
- Always use forward slashes in paths (C:/Users/... not C:\\Users\\...)
- Only use ONE tool call at a time, then wait for the result
- After getting a tool result, continue your work or call another tool
- When you're done, give your final answer WITHOUT any [TOOL_CALL] tags
"""


@dataclass
class ToolResult:
    """Result from executing a tool."""
    tool: str
    success: bool
    output: str
    error: Optional[str] = None


def read_file(path: str) -> ToolResult:
    """Read contents of a file."""
    try:
        p = Path(path)
        if not p.exists():
            return ToolResult(tool="read_file", success=False, output="", error=f"File not found: {path}")
        if not p.is_file():
            return ToolResult(tool="read_file", success=False, output="", error=f"Not a file: {path}")

        content = p.read_text(encoding='utf-8', errors='replace')
        if len(content) > 15000:
            content = content[:15000] + "\n\n... [truncated — file too large, showing first 15000 chars]"

        return ToolResult(tool="read_file", success=True, output=content)
    except Exception as e:
        return ToolResult(tool="read_file", success=False, output="", error=str(e))


def write_file(path: str, content: str) -> ToolResult:
    """Write content to a file, creating parent directories if needed."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding='utf-8')
        return ToolResult(tool="write_file", success=True, output=f"Successfully wrote {len(content)} chars to {path}")
    except Exception as e:
        return ToolResult(tool="write_file", success=False, output="", error=str(e))


def run_command(command: str, timeout: int = 30) -> ToolResult:
    """Run a shell command and return output."""
    # Block dangerous commands
    dangerous = ['rm -rf /', 'format', 'del /s /q C:', 'shutdown', 'mkfs']
    if any(d in command.lower() for d in dangerous):
        return ToolResult(tool="run_command", success=False, output="", error="Blocked: dangerous command detected")

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(Path.home()),
        )

        output = result.stdout
        if result.stderr:
            output += "\n[STDERR]\n" + result.stderr

        if len(output) > 10000:
            output = output[:10000] + "\n... [truncated]"

        return ToolResult(
            tool="run_command",
            success=result.returncode == 0,
            output=output or "(no output)",
            error=f"Exit code: {result.returncode}" if result.returncode != 0 else None,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(tool="run_command", success=False, output="", error=f"Command timed out after {timeout}s")
    except Exception as e:
        return ToolResult(tool="run_command", success=False, output="", error=str(e))


def list_dir(path: str) -> ToolResult:
    """List directory contents."""
    try:
        p = Path(path)
        if not p.exists():
            return ToolResult(tool="list_dir", success=False, output="", error=f"Directory not found: {path}")
        if not p.is_dir():
            return ToolResult(tool="list_dir", success=False, output="", error=f"Not a directory: {path}")

        entries = []
        for item in sorted(p.iterdir()):
            if item.name.startswith('.'):
                continue
            if item.is_dir():
                count = sum(1 for _ in item.iterdir()) if item.exists() else 0
                entries.append(f"📁 {item.name}/  ({count} items)")
            else:
                size = item.stat().st_size
                if size > 1024 * 1024:
                    size_str = f"{size / 1024 / 1024:.1f} MB"
                elif size > 1024:
                    size_str = f"{size / 1024:.1f} KB"
                else:
                    size_str = f"{size} B"
                entries.append(f"📄 {item.name}  ({size_str})")

        if not entries:
            return ToolResult(tool="list_dir", success=True, output="(empty directory)")

        return ToolResult(tool="list_dir", success=True, output="\n".join(entries[:100]))
    except Exception as e:
        return ToolResult(tool="list_dir", success=False, output="", error=str(e))


def search_files(path: str, pattern: str) -> ToolResult:
    """Search for files matching a glob pattern."""
    try:
        p = Path(path)
        if not p.exists():
            return ToolResult(tool="search_files", success=False, output="", error=f"Path not found: {path}")

        matches = list(p.rglob(pattern))[:50]  # Limit results

        if not matches:
            return ToolResult(tool="search_files", success=True, output=f"No files matching '{pattern}' found in {path}")

        result = "\n".join(str(m) for m in matches)
        return ToolResult(tool="search_files", success=True, output=result)
    except Exception as e:
        return ToolResult(tool="search_files", success=False, output="", error=str(e))


def read_url(url: str) -> ToolResult:
    """Fetch and extract text from a URL."""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode('utf-8', errors='ignore')
            
        # Strip HTML tags
        text = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        text = '\n'.join(line for line in lines if line)
        
        if len(text) > 15000:
            text = text[:15000] + "\n... [truncated]"
            
        return ToolResult(tool="read_url", success=True, output=text)
    except Exception as e:
        return ToolResult(tool="read_url", success=False, output="", error=str(e))


def search_web(query: str) -> ToolResult:
    """Search the web using DuckDuckGo HTML."""
    try:
        encoded_query = urllib.parse.quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
        
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode('utf-8', errors='ignore')
            
        # Very basic parsing of DDG HTML
        results = []
        snippets = re.findall(r'<a class="result__url" href="([^"]+)">(.*?)</a>.*?<a class="result__snippet[^>]*>(.*?)</a>', html, re.DOTALL | re.IGNORECASE)
        
        for i, (link, _, snippet) in enumerate(snippets[:5]):
            # Clean snippet html
            clean_snippet = re.sub(r'<[^>]+>', '', snippet).strip()
            # Clean redirect link if present
            if 'uddg=' in link:
                try:
                    link = urllib.parse.unquote(link.split('uddg=')[1].split('&')[0])
                except:
                    pass
            results.append(f"Result {i+1}:\nURL: {link}\nSnippet: {clean_snippet}\n")
            
        if not results:
            return ToolResult(tool="search_web", success=True, output="No results found.")
            
        return ToolResult(tool="search_web", success=True, output="\n".join(results))
    except Exception as e:
        return ToolResult(tool="search_web", success=False, output="", error=str(e))


# ─── Tool Registry ──────────────────────────────────────────────────────────

TOOL_FUNCTIONS = {
    "read_file": lambda args: read_file(args.get("path", "")),
    "write_file": lambda args: write_file(args.get("path", ""), args.get("content", "")),
    "run_command": lambda args: run_command(args.get("command", ""), args.get("timeout", 30)),
    "list_dir": lambda args: list_dir(args.get("path", "")),
    "search_files": lambda args: search_files(args.get("path", ""), args.get("pattern", "")),
    "search_web": lambda args: search_web(args.get("query", "")),
    "read_url": lambda args: read_url(args.get("url", "")),
}


def execute_tool(tool_name: str, args: dict) -> ToolResult:
    """Execute a tool by name with given arguments."""
    if tool_name not in TOOL_FUNCTIONS:
        return ToolResult(tool=tool_name, success=False, output="", error=f"Unknown tool: {tool_name}")

    logger.info(f"  🔧 Executing tool: {tool_name}({args})")
    result = TOOL_FUNCTIONS[tool_name](args)

    if result.success:
        logger.info(f"  ✓ Tool succeeded: {len(result.output)} chars")
    else:
        logger.warning(f"  ✗ Tool failed: {result.error}")

    return result
