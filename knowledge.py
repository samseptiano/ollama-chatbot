# knowledge.py
import json
import os
from datetime import datetime
from typing import List, Dict, Optional
import uuid

KNOWLEDGE_FILE = "knowledge.json"

def load_knowledge() -> List[Dict]:
    if not os.path.exists(KNOWLEDGE_FILE):
        return []
    try:
        with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []


def save_knowledge(data: List[Dict]):
    try:
        with open(KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving knowledge.json: {e}")


def add_to_knowledge(session_id: str, role: str, content: str, tool_name: Optional[str] = None):
    history = load_knowledge()

    entry = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "role": role,
        "content": content
    }
    if tool_name:
        entry["tool_name"] = tool_name

    history.append(entry)

    if len(history) > 1000:
        history = history[-700:]

    save_knowledge(history)


def get_history(session_id: str, limit: int = 15) -> List[Dict]:
    history = load_knowledge()
    filtered = [msg for msg in history if msg.get("session_id") == session_id]
    return filtered[-limit:]


def generate_session_id() -> str:
    return str(uuid.uuid4())
