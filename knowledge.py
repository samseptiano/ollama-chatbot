# knowledge.py - Menggunakan ChromaDB
import uuid
from datetime import datetime
from typing import List, Dict, Optional

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# ====================== KONFIGURASI ======================
CHROMA_PERSIST_DIR = "./chroma_chat_db"
EMBEDDING_MODEL = "mxbai-embed-large"        # Model terbaik saat ini

# Inisialisasi embeddings
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

vectorstore = Chroma(
    persist_directory=CHROMA_PERSIST_DIR,
    embedding_function=embeddings,
    collection_name="chat_history"
)


def add_to_knowledge(session_id: str, role: str, content: str, tool_name: Optional[str] = None):
    """Tambahkan pesan ke ChromaDB"""
    text = f"[{role.upper()}] {content}"

    metadata = {
        "session_id": session_id,
        "role": role,
        "timestamp": datetime.now().isoformat(),
    }
    if tool_name:
        metadata["tool_name"] = tool_name

    vectorstore.add_texts(
        texts=[text],
        metadatas=[metadata],
        ids=[str(uuid.uuid4())]
    )


def get_history(session_id: str, limit: int = 15) -> List[Dict]:
    """Ambil history percakapan berdasarkan session_id"""
    try:
        results = vectorstore.similarity_search(
            query="previous conversation context",
            k=limit,
            filter={"session_id": session_id}
        )

        history = []
        for doc in results:
            content = doc.page_content
            # Bersihkan prefix [ROLE]
            for prefix in ["[USER] ", "[ASSISTANT] ", "[TOOL] "]:
                if content.startswith(prefix):
                    content = content[len(prefix):]
                    break

            history.append({
                "role": doc.metadata.get("role", "assistant"),
                "content": content,
                "tool_name": doc.metadata.get("tool_name")
            })

        return history

    except Exception as e:
        print(f"Error retrieving history from ChromaDB: {e}")
        return []


def generate_session_id() -> str:
    return str(uuid.uuid4())


def get_collection_info():
    """Melihat jumlah dokumen di database"""
    try:
        count = vectorstore._collection.count()
        print(f"Total documents in ChromaDB: {count}")
        return count
    except:
        return 0
