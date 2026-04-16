# chatbot.py
import ollama
import requests
import json
from typing import List, Dict, Optional
from knowledge import add_to_knowledge, get_history, generate_session_id

# ====================== TOOLS ======================
def fetch_users(name: Optional[str] = None, limit: int = 8):
    """
    Fetch users data from JSONPlaceholder.

    Use this tool ONLY when the user asks about:
    - User details, address, email, phone, company, or personal information.
    - Names either first name or last name like "Patricia", "Leanne", "Ervin", "Graham", etc.
    - Questions such as: "show me Patricia's address", "where does Patricia live?",
      "details of Patricia", "Patricia info", "user list", etc.

    Do NOT guess or ask for more information. Always call this tool if a name is mentioned.
    """
    try:
        r = requests.get("https://jsonplaceholder.typicode.com/users", timeout=10)
        r.raise_for_status()
        users = r.json()
        if name:
            name_lower = name.lower().strip()
            users = [u for u in users if name_lower in u.get("name", "").lower()]
        users = users[:limit]
        return json.dumps(users, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error fetching users: {str(e)}"


def fetch_posts(limit: int = 10):
    try:
        r = requests.get("https://jsonplaceholder.typicode.com/posts", timeout=10)
        r.raise_for_status()
        return json.dumps(r.json()[:limit], ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error fetching posts: {str(e)}"


def fetch_products(limit: int = 12):
    try:
        r = requests.get("https://fakestoreapi.com/products", timeout=10)
        r.raise_for_status()
        return json.dumps(r.json()[:limit], ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error fetching products: {str(e)}"


def fetch_pokemon(pokemon: str):
    try:
        r = requests.get(f"https://pokeapi.co/api/v2/pokemon/{pokemon.lower()}", timeout=10)
        r.raise_for_status()
        data = r.json()
        return json.dumps({
            "name": data.get("name"),
            "types": [t["type"]["name"] for t in data.get("types", [])],
            "height": data.get("height"),
            "weight": data.get("weight"),
            "abilities": data.get("abilities")
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error fetching pokemon '{pokemon}': {str(e)}"


tools_list = [fetch_users, fetch_posts, fetch_products, fetch_pokemon]


# ====================== MAIN FUNCTION ======================
def get_chat_response(
    message: str,
    user_id: str,
    session_id: Optional[str] = None,
    new_conversation: bool = False
) -> dict:

    # Logika pembuatan session_id
    if new_conversation or session_id is None or session_id == "":
        session_id = generate_session_id()
        print(f"[NEW CONVERSATION] Created new session_id: {session_id} for user: {user_id}")
    else:
        print(f"[CONTINUE CONVERSATION] Using session_id: {session_id} for user: {user_id}")

    # Ambil history percakapan
    history = get_history(session_id, limit=12)

    system_prompt = {
        "role": "system",
        "content": f"""You are a friendly and helpful English-speaking assistant.

Conversation Memory (Session ID: {session_id} | User ID: {user_id})
Maintain context from previous messages in this session.
If user says "nya", "beratnya", "tingginya", "it", etc., refer to the last mentioned topic.

Rules:
- Greet normally without tools.
- Use tools only when necessary.
- Be contextual and consistent with previous messages.
"""
    }

    messages = [system_prompt] + history
    messages.append({"role": "user", "content": message})

    try:
        response = ollama.chat(
            model="qwen2.5:7b",
            messages=messages,
            tools=tools_list,
            stream=False,
            options={"temperature": 0.3, "num_ctx": 8192}
        )

        assistant_message = response['message']
        tool_calls = assistant_message.get('tool_calls')

        if tool_calls:
            for tool_call in tool_calls:
                func_name = tool_call['function']['name']
                args = tool_call['function'].get('arguments', {}) or {}

                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except:
                        args = {}

                tool_func = next((t for t in tools_list if t.__name__ == func_name), None)

                if tool_func:
                    try:
                        result = tool_func(**args) if func_name == "fetch_pokemon" else tool_func()
                    except Exception as e:
                        result = f"Error: {str(e)}"
                else:
                    result = "Tool not found"

                add_to_knowledge(session_id, "tool", str(result), tool_name=func_name)
                messages.append({"role": "tool", "content": str(result), "tool_name": func_name})

            # Second pass
            final_response = ollama.chat(
                model="qwen2.5:7b",
                messages=messages,
                stream=False,
                options={"temperature": 0.7}
            )
            reply = final_response['message']['content']
        else:
            reply = assistant_message['content']

        # Simpan ke knowledge.json
        add_to_knowledge(session_id, "user", message)
        add_to_knowledge(session_id, "assistant", reply)

        return {
            "reply": reply,
            "user_id": user_id,
            "session_id": session_id,
            "new_conversation": new_conversation
        }

    except Exception as e:
        error_msg = f"Sorry, an error occurred: {str(e)}"
        add_to_knowledge(session_id, "assistant", error_msg)
        return {
            "reply": error_msg,
            "user_id": user_id,
            "session_id": session_id,
            "new_conversation": new_conversation
        }
