"""
config.py — Central configuration for the RAG Chatbot.
All paths, defaults, and env variables in one place.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
VECTOR_STORES_DIR = DATA_DIR / "vector_stores"
DB_PATH = DATA_DIR / "chat_history.db"
UPLOADS_DIR = DATA_DIR / "uploads"

for _dir in [DATA_DIR, VECTOR_STORES_DIR, UPLOADS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── LLM ────────────────────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")          # "openai" | "ollama"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# ── Embeddings ─────────────────────────────────────────────────────────
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")   # "openai" | "local"
LOCAL_EMBEDDING_MODEL = os.getenv(
    "LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# ── RAG ────────────────────────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_DOCS = 5
# Relevance threshold (0–1). Below this → bot says "I don't know".
RELEVANCE_THRESHOLD = 0.30
# How many last message-pairs to include as conversation context
MAX_HISTORY_PAIRS = 4

# ── UI defaults ────────────────────────────────────────────────────────
DEFAULT_BOT_NAME = "Ассистент"
DEFAULT_SYSTEM_PROMPT = (
    "Ты — полезный ассистент. Отвечай строго на основе предоставленных документов. "
    "Если информации нет — честно скажи об этом."
)
NO_CONTEXT_REPLY = (
    "❌ Я не нашёл в документах информации по этому вопросу. "
    "Попробуйте переформулировать или загрузите нужные документы."
)
