"""
Token-Efficient Routing Agent — Web Interface

A beautiful chat app powered by your local Ollama models.
Supports text chat, file uploads, and image attachments.

Usage:
    python app.py
    Then open http://localhost:5000 in your browser.
"""

import os
import io
import sys
import json
import base64
import time
import logging
from pathlib import Path

# Force UTF-8
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from flask import Flask, request, jsonify, render_template, session
from config import load_config
from classifier import classify
from multi_agent import AgenticSystem
from providers import CompletionResult

app = Flask(__name__)
app.secret_key = "super_secret_token_router_key_for_local_dev"

# ─── Global State ────────────────────────────────────────────────────────────
config = load_config()
agent_sys = AgenticSystem(config)

SYSTEM_PROMPT = (
    "You are a helpful, knowledgeable AI assistant. "
    "Answer accurately and concisely. "
    "If the user uploads a file, analyze its contents carefully. "
    "Maintain context from the conversation history."
)

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

MAX_HISTORY_TURNS = 10

# Persistent conversation store
CHATS_FILE = Path(__file__).parent / "chats.json"

def load_chats():
    if CHATS_FILE.exists():
        try:
            return json.loads(CHATS_FILE.read_text(encoding='utf-8'))
        except:
            pass
    return {}

def save_chats():
    CHATS_FILE.write_text(json.dumps(conversations, indent=2), encoding='utf-8')

conversations = load_chats()

# Allowed file extensions
ALLOWED_EXTENSIONS = {
    'txt', 'py', 'js', 'ts', 'html', 'css', 'json', 'md', 'csv',
    'java', 'cpp', 'c', 'h', 'rs', 'go', 'rb', 'php', 'sql', 'yaml', 'yml',
    'xml', 'sh', 'bat', 'log', 'cfg', 'ini', 'toml',
    'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp',
    'pdf', 'doc', 'docx',
}

IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'}


def get_file_extension(filename: str) -> str:
    return filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''


def read_text_file(filepath: Path) -> str:
    """Read text content from a file, truncating if too long."""
    try:
        content = filepath.read_text(encoding='utf-8', errors='replace')
        if len(content) > 8000:
            content = content[:8000] + "\n\n... [truncated — file is very large]"
        return content
    except Exception as e:
        return f"[Error reading file: {e}]"


def get_conversation(session_id: str) -> dict:
    """Get or create conversation history for a session."""
    if session_id not in conversations:
        conversations[session_id] = {
            "id": session_id,
            "title": "New Chat",
            "updated_at": time.time(),
            "messages": []
        }
        save_chats()
    return conversations[session_id]


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the chat interface."""
    if "session_id" not in session:
        session["session_id"] = os.urandom(16).hex()
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """Process a chat message with optional file context."""
    data = request.get_json()
    user_message = data.get("message", "").strip()
    file_context = data.get("file_context", "")
    image_b64 = data.get("image_b64", "")

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    session_id = data.get("session_id") or session.get("session_id", "default")
    session["session_id"] = session_id
    
    chat_session = get_conversation(session_id)
    history = chat_session["messages"]

    # Build the full prompt
    full_prompt = user_message
    if file_context:
        full_prompt = (
            f"The user has uploaded a file with the following contents:\n"
            f"```\n{file_context}\n```\n\n"
            f"User's message: {user_message}"
        )

    # Classify the task
    profile = classify(full_prompt)

    # Route and get result
    start_time = time.perf_counter()
    result = agent_sys.run_task(
        prompt=full_prompt,
        history=history,
    )
    elapsed = (time.perf_counter() - start_time) * 1000

    # Save to history
    if result.success and result.text:
        # Generate a title if it's the first message
        if not history and user_message:
            chat_session["title"] = user_message[:30] + ("..." if len(user_message) > 30 else "")
            
        chat_session["messages"].append({"role": "user", "content": full_prompt})
        chat_session["messages"].append({"role": "assistant", "content": result.text})
        chat_session["updated_at"] = time.time()
        
        # Trim history
        if len(chat_session["messages"]) > MAX_HISTORY_TURNS * 2:
            chat_session["messages"] = chat_session["messages"][-(MAX_HISTORY_TURNS * 2):]
            
        save_chats()

    return jsonify({
        "answer": result.text if result.success else "Sorry, I couldn't process that. Please try again.",
        "model": result.model_id,
        "provider": result.provider,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "latency_ms": round(elapsed),
        "difficulty": profile.difficulty.name,
        "success": result.success,
    })


@app.route("/api/upload", methods=["POST"])
def upload_file():
    """Handle file upload and return extracted content."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No filename"}), 400

    ext = get_file_extension(file.filename)
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"File type .{ext} not supported"}), 400

    # Save file
    safe_name = f"{int(time.time())}_{file.filename}"
    filepath = UPLOAD_DIR / safe_name
    file.save(filepath)

    response = {
        "filename": file.filename,
        "type": "image" if ext in IMAGE_EXTENSIONS else "text",
        "size": filepath.stat().st_size,
    }

    if ext in IMAGE_EXTENSIONS:
        # Read as base64 for display and potential vision model use
        with open(filepath, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        response["image_b64"] = b64
        response["content"] = f"[Image: {file.filename}]"
    else:
        # Read text content
        content = read_text_file(filepath)
        response["content"] = content

    return jsonify(response)


@app.route("/api/sessions", methods=["GET"])
def list_sessions():
    """List all chat sessions."""
    sessions = []
    for sid, data in conversations.items():
        sessions.append({
            "id": sid,
            "title": data.get("title", "New Chat"),
            "updated_at": data.get("updated_at", 0)
        })
    # Sort by updated_at descending
    sessions.sort(key=lambda x: x["updated_at"], reverse=True)
    return jsonify(sessions)


@app.route("/api/sessions/<session_id>", methods=["GET"])
def get_session(session_id):
    """Get a specific session."""
    if session_id in conversations:
        session["session_id"] = session_id
        return jsonify(conversations[session_id])
    return jsonify({"error": "Not found"}), 404


@app.route("/api/sessions", methods=["POST"])
def new_session():
    """Create a new session."""
    session_id = os.urandom(16).hex()
    session["session_id"] = session_id
    conversations[session_id] = {
        "id": session_id,
        "title": "New Chat",
        "updated_at": time.time(),
        "messages": []
    }
    save_chats()
    return jsonify({"id": session_id})


@app.route("/api/sessions/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    """Delete a session."""
    if session_id in conversations:
        del conversations[session_id]
        save_chats()
        
        if session.get("session_id") == session_id:
            # Switch to a new session if we deleted the active one
            if conversations:
                session["session_id"] = list(conversations.keys())[-1]
            else:
                session["session_id"] = os.urandom(16).hex()
                
        return jsonify({"success": True})
    return jsonify({"error": "Not found"}), 404


@app.route("/api/clear", methods=["POST"])
def clear_conversation():
    """Clear conversation history for current session."""
    session_id = session.get("session_id", "default")
    if session_id in conversations:
        conversations[session_id]["messages"] = []
        conversations[session_id]["title"] = "New Chat"
        save_chats()
    return jsonify({"status": "cleared"})


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Get token usage stats."""
    tracker = agent_sys.router.tracker
    return jsonify({
        "fireworks_tokens": tracker.total_fireworks_tokens,
        "local_tokens": tracker.total_local_tokens,
        "fireworks_calls": tracker.fireworks_calls,
        "local_calls": tracker.local_calls,
    })


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("\n  🌐 Token Router Web App starting...")
    print("  📍 Open http://localhost:5000 in your browser\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
