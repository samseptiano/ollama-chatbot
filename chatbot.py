import ollama
import requests
import json

# ====================== TOOLS DEFINITION ======================
def fetch_users():
    """Fetch list of fake users from JSONPlaceholder"""
    try:
        r = requests.get("https://jsonplaceholder.typicode.com/users", timeout=10)
        r.raise_for_status()
        return json.dumps(r.json(), ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error fetching users: {str(e)}"


def fetch_posts():
    """Fetch list of fake posts from JSONPlaceholder (limit to 10)"""
    try:
        r = requests.get("https://jsonplaceholder.typicode.com/posts", timeout=10)
        r.raise_for_status()
        return json.dumps(r.json()[:10], ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error fetching posts: {str(e)}"


def fetch_products():
    """Fetch list of products from Fake Store API"""
    try:
        r = requests.get("https://fakestoreapi.com/products", timeout=10)
        r.raise_for_status()
        return json.dumps(r.json(), ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error fetching products: {str(e)}"


def fetch_pokemon(pokemon: str):
    """Fetch Pokemon information from PokeAPI"""
    try:
        r = requests.get(f"https://pokeapi.co/api/v2/pokemon/{pokemon.lower()}", timeout=10)
        r.raise_for_status()
        data = r.json()
        return json.dumps({
            "name": data.get("name"),
            "types": [t["type"]["name"] for t in data.get("types", [])],
            "height": data.get("height"),
            "weight": data.get("weight")
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error fetching pokemon '{pokemon}': {str(e)}"


tools_list = [fetch_users, fetch_posts, fetch_products, fetch_pokemon]

# ====================== MAIN CHATBOT ======================
def run_chatbot():
    print("🤖 Ollama Multi-API Chatbot Ready!")
    print("Type 'exit' to quit.\n")

    messages = [{
        "role": "system",
        "content": """You are a friendly English-speaking assistant.

Rules:
- For normal greetings like "Hi", "Hello", "How are you?", "Hey", answer directly and friendly. Do NOT call any tools.
- Only call tools when the user asks for data about:
  • Users or posts → use fetch_users or fetch_posts
  • Products in the store → use fetch_products
  • Pokémon → use fetch_pokemon with the pokemon name
"""
    }]

    while True:
        user_input = input("👤 You: ").strip()
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("👋 Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        print("🤖 Bot: ", end="", flush=True)

        # First pass: Get response + possible tool calls
        response_stream = ollama.chat(
            model="qwen2.5:3b",          # Change to "qwen2.5:7b" if you have more RAM
            messages=messages,
            tools=tools_list,
            stream=True,
            options={"temperature": 0.7, "num_ctx": 8192}
        )

        full_content = ""
        tool_calls = []

        for chunk in response_stream:
            if chunk.message.content:
                print(chunk.message.content, end="", flush=True)
                full_content += chunk.message.content

            if chunk.message.tool_calls:
                tool_calls.extend(chunk.message.tool_calls)

        print()  # new line

        # Save assistant message
        assistant_msg = {"role": "assistant", "content": full_content}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        # ==================== EXECUTE TOOLS ====================
        if tool_calls:
            print("   🔧 Executing tool...")

            for tool_call in tool_calls:
                func_name = tool_call.function.name
                args = tool_call.function.arguments or {}

                print(f"      → Calling {func_name} {args}")

                # Find and run the tool
                tool_func = next((t for t in tools_list if t.__name__ == func_name), None)

                if tool_func:
                    try:
                        if func_name == "fetch_pokemon":
                            result = tool_func(**args)
                        else:
                            result = tool_func()
                    except Exception as e:
                        result = f"Error executing tool: {e}"
                else:
                    result = "Tool not found"

                # Add tool result to conversation history
                messages.append({
                    "role": "tool",
                    "tool_name": func_name,
                    "content": str(result)
                })

            # Second pass: Ask the model to generate final answer using tool results
            print("🤖 Bot: ", end="", flush=True)

            final_stream = ollama.chat(
                model="qwen2.5:3b",
                messages=messages,
                stream=True,
                options={"temperature": 0.7}
            )

            for chunk in final_stream:
                if chunk.message.content:
                    print(chunk.message.content, end="", flush=True)

            print("\n")


if __name__ == "__main__":
    run_chatbot()
