## Requirement: 
```sh
Python 3.11.4
```

Knowledge source API (sample only)
```sh
https://jsonplaceholder.typicode.com/users
https://jsonplaceholder.typicode.com/posts
https://fakestoreapi.com/products
https://pokeapi.co/api/v2/pokemon/[pokemon_name]
```
### 1. Install Ollama
Download & install from https://ollama.com (Windows/Mac/Linux).
Open terminal and run:
```sh
ollama serve
```

### 2. Pull supported tool calling model (important!)
Recommendation in 2026:
```sh
ollama pull qwen2.5:7b  # lightweight &  best for tool calling
```
or
```sh
ollama pull llama3.2:latest  # if prefer Llama
```
### 3. Create project folder & virtual environment
```sh
mkdir ollama-multi-api-chatbot
cd ollama-multi-api-chatbot
python -m venv venv

# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate
```


### 4. Install library Python
```sh
pip install ollama requests
```

### 5. Setup chatbot.py files

### 6. To run app, run in bash/cmd
```sh
python chatbot.py
```

<img width="1500" height="500" alt="Screenshot 2026-04-09 131211" src="https://github.com/user-attachments/assets/d66aa9c8-5e97-42e8-9a10-bd3fb1cebb4c" />

