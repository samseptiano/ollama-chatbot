import ollama
import requests
import json

# ====================== DEFINISI TOOLS ======================
def fetch_users():
    """Ambil daftar user"""
    try:
        r = requests.get("https://jsonplaceholder.typicode.com/users", timeout=10)
        r.raise_for_status()
        return json.dumps(r.json(), ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error users: {str(e)}"


def fetch_posts():
    """Ambil daftar posts"""
    try:
        r = requests.get("https://jsonplaceholder.typicode.com/posts", timeout=10)
        r.raise_for_status()
        return json.dumps(r.json()[:10], ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error posts: {str(e)}"


def fetch_products():
    """Ambil daftar produk dari Fake Store API"""
    try:
        r = requests.get("https://fakestoreapi.com/products", timeout=10)
        r.raise_for_status()
        return json.dumps(r.json(), ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error products: {str(e)}"


def fetch_pokemon(pokemon: str):
    """Ambil info pokemon"""
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
        return f"Error pokemon: {str(e)}"


tools_list = [fetch_users, fetch_posts, fetch_products, fetch_pokemon]

# ====================== CHATBOT ======================
def run_chatbot():
    print("🤖 Chatbot Ollama Multi API Siap!")
    print("Ketik 'exit' untuk keluar.\n")

    messages = [{
        "role": "system",
        "content": """Kamu asisten ramah berbahasa Indonesia.
Untuk sapaan biasa (Hai, Halo, Apa kabar) jawab langsung tanpa tool.
Hanya panggil tool jika user meminta data tentang:
- users atau posts → gunakan fetch_users / fetch_posts
- produk di toko → gunakan fetch_products
- pokemon → gunakan fetch_pokemon"""
    }]

    while True:
        user_input = input("👤 Kamu: ").strip()
        if user_input.lower() in ["exit", "keluar", "quit", "bye"]:
            print("👋 Sampai jumpa!")
            break

        messages.append({"role": "user", "content": user_input})

        print("🤖 Bot: ", end="", flush=True)

        # Streaming response
        response_stream = ollama.chat(
            model="qwen2.5:3b",          # Pakai 3b dulu biar cepat
            messages=messages,
            tools=tools_list,
            stream=True,
            options={"temperature": 0.7, "num_ctx": 8192}
        )

        full_content = ""
        tool_calls = []

        for chunk in response_stream:
            # Tampilkan teks yang keluar
            if chunk.message.content:
                print(chunk.message.content, end="", flush=True)
                full_content += chunk.message.content

            # Kumpulkan tool calls
            if chunk.message.tool_calls:
                tool_calls.extend(chunk.message.tool_calls)

        print()  # baris baru

        # Simpan respons assistant
        assistant_msg = {"role": "assistant", "content": full_content}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        # ==================== EKSEKUSI TOOL ====================
        if tool_calls:
            print("   🔧 Menjalankan tool...")

            for tool_call in tool_calls:
                func_name = tool_call.function.name
                args = tool_call.function.arguments or {}

                print(f"      → Memanggil {func_name} {args}")

                if func_name in [t.__name__ for t in tools_list]:
                    # Cari function yang sesuai
                    tool_func = next(t for t in tools_list if t.__name__ == func_name)

                    try:
                        if func_name == "fetch_pokemon":
                            result = tool_func(**args)
                        else:
                            result = tool_func()
                    except Exception as e:
                        result = f"Error saat menjalankan tool: {e}"

                    # Tambahkan hasil tool ke messages
                    messages.append({
                        "role": "tool",
                        "tool_name": func_name,
                        "content": str(result)
                    })

            # Setelah tool selesai, minta model jawab lagi dengan data tool
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