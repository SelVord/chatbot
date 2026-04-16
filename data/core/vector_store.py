"""
core/vector_store.py — Create, update, save, and load FAISS vector stores.

Each session keeps its own FAISS index on disk so embeddings survive restarts.
"""

from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

import config
from core.embeddings import get_embeddings


def _store_path(session_id: int) -> Path:
    return config.VECTOR_STORES_DIR / str(session_id)


def build_vector_store(
    session_id: int,
    chunks: list[Document],
    embedding_provider: str = config.EMBEDDING_PROVIDER,
) -> FAISS:
    """Create a new FAISS index from chunks and save to disk."""
    embeddings = get_embeddings(embedding_provider)
    store = FAISS.from_documents(chunks, embeddings)
    store_path = _store_path(session_id)
    store.save_local(str(store_path))
    return store


def add_to_vector_store(
    session_id: int,
    chunks: list[Document],
    embedding_provider: str = config.EMBEDDING_PROVIDER,
) -> FAISS:
    """
    Add new chunks to an existing FAISS index.
    Creates the index if it doesn't exist yet.
    """
    store_path = _store_path(session_id)
    embeddings = get_embeddings(embedding_provider)

    if store_path.exists():
        store = FAISS.load_local(
            str(store_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        store.add_documents(chunks)
    else:
        store = FAISS.from_documents(chunks, embeddings)

    store.save_local(str(store_path))
    return store


def load_vector_store(
    session_id: int,
    embedding_provider: str = config.EMBEDDING_PROVIDER,
) -> Optional[FAISS]:
    """Load FAISS index from disk. Returns None if not found."""
    store_path = _store_path(session_id)
    if not store_path.exists():
        return None
    embeddings = get_embeddings(embedding_provider)
    return FAISS.load_local(
        str(store_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def delete_vector_store(session_id: int) -> None:
    """Delete the FAISS index directory for a session."""
    import shutil
    store_path = _store_path(session_id)
    if store_path.exists():
        shutil.rmtree(store_path)


def has_vector_store(session_id: int) -> bool:
    return _store_path(session_id).exists()
