"""
core/database.py — SQLite-based storage for sessions and chat history.

Schema:
  sessions  — one row per chatbot instance (name, config, vector store path)
  messages  — all messages, linked to a session
  documents — metadata about uploaded PDFs per session
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import config


# ── Connection helper ──────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(config.DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ── Init ───────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create all tables if they don't exist."""
    conn = _connect()
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            name              TEXT    NOT NULL,
            bot_name          TEXT    NOT NULL DEFAULT 'Ассистент',
            system_prompt     TEXT    NOT NULL DEFAULT '',
            vector_store_path TEXT    NOT NULL DEFAULT '',
            llm_provider      TEXT    NOT NULL DEFAULT 'openai',
            llm_model         TEXT    NOT NULL DEFAULT 'gpt-4o-mini',
            embedding_provider TEXT   NOT NULL DEFAULT 'openai',
            created_at        TEXT    NOT NULL,
            updated_at        TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            role        TEXT    NOT NULL,   -- 'user' | 'assistant'
            content     TEXT    NOT NULL,
            sources     TEXT    NOT NULL DEFAULT '[]',   -- JSON list of source dicts
            created_at  TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS documents (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            filename    TEXT    NOT NULL,
            page_count  INTEGER NOT NULL DEFAULT 0,
            chunk_count INTEGER NOT NULL DEFAULT 0,
            uploaded_at TEXT    NOT NULL
        );
    """)

    conn.commit()
    conn.close()


# ── Sessions ───────────────────────────────────────────────────────────

def create_session(
    name: str,
    bot_name: str = config.DEFAULT_BOT_NAME,
    system_prompt: str = config.DEFAULT_SYSTEM_PROMPT,
    llm_provider: str = config.LLM_PROVIDER,
    llm_model: str = config.OPENAI_MODEL,
    embedding_provider: str = config.EMBEDDING_PROVIDER,
) -> int:
    now = datetime.utcnow().isoformat()
    conn = _connect()
    cur = conn.execute(
        """INSERT INTO sessions
               (name, bot_name, system_prompt, vector_store_path,
                llm_provider, llm_model, embedding_provider, created_at, updated_at)
           VALUES (?, ?, ?, '', ?, ?, ?, ?, ?)""",
        (name, bot_name, system_prompt, llm_provider, llm_model, embedding_provider, now, now),
    )
    session_id = cur.lastrowid
    conn.commit()
    conn.close()
    return session_id


def get_session(session_id: int) -> Optional[dict]:
    conn = _connect()
    row = conn.execute(
        "SELECT * FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def list_sessions() -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM sessions ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_session(session_id: int, **fields) -> None:
    fields["updated_at"] = datetime.utcnow().isoformat()
    sets = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [session_id]
    conn = _connect()
    conn.execute(f"UPDATE sessions SET {sets} WHERE id = ?", values)
    conn.commit()
    conn.close()


def delete_session(session_id: int) -> None:
    conn = _connect()
    conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()


# ── Messages ───────────────────────────────────────────────────────────

def save_message(
    session_id: int,
    role: str,
    content: str,
    sources: list[dict] | None = None,
) -> int:
    now = datetime.utcnow().isoformat()
    conn = _connect()
    cur = conn.execute(
        "INSERT INTO messages (session_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?)",
        (session_id, role, content, json.dumps(sources or []), now),
    )
    msg_id = cur.lastrowid
    conn.commit()
    conn.close()
    return msg_id


def get_messages(session_id: int) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at",
        (session_id,),
    ).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["sources"] = json.loads(d["sources"])
        result.append(d)
    return result


def clear_messages(session_id: int) -> None:
    conn = _connect()
    conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()


def export_messages_json(session_id: int) -> str:
    """Return chat history as a JSON string for download."""
    session = get_session(session_id)
    messages = get_messages(session_id)
    export = {
        "session": session,
        "messages": messages,
        "exported_at": datetime.utcnow().isoformat(),
    }
    return json.dumps(export, ensure_ascii=False, indent=2)


# ── Documents ──────────────────────────────────────────────────────────

def save_document(
    session_id: int,
    filename: str,
    page_count: int = 0,
    chunk_count: int = 0,
) -> None:
    now = datetime.utcnow().isoformat()
    conn = _connect()
    conn.execute(
        """INSERT INTO documents (session_id, filename, page_count, chunk_count, uploaded_at)
           VALUES (?, ?, ?, ?, ?)""",
        (session_id, filename, page_count, chunk_count, now),
    )
    conn.commit()
    conn.close()


def get_documents(session_id: int) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM documents WHERE session_id = ? ORDER BY uploaded_at",
        (session_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
