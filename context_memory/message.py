"""
context_memory/message.py
Environment variables expected (can also come from config.py / .env):
  POSTGRES_HOST        default: localhost
  POSTGRES_PORT        default: 5432
  POSTGRES_DB          default: ragdb
  POSTGRES_USER        default: postgres
  POSTGRES_PASSWORD    (required)
  OLLAMA_BASE_URL      default: http://localhost:11434
  OLLAMA_EMBED_MODEL   default: nomic-embed-text
  MEMO_COLLECTION      default: rag_chat_memory
  MAX_HISTORY_MESSAGES default: 6      (pairs kept per session)
  MAX_MSG_CHARS        default: 300    (chars kept per message)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence
from config import settings
import urllib.parse

import psycopg2
import psycopg2.extras
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    messages_from_dict,
    messages_to_dict,
)
from mem0 import Memory

logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────


PG_HOST     = settings.PG_HOST
PG_PORT     = settings.PG_PORT
PG_DB       = settings.PG_DB
PG_USER     = settings.PG_USER
PG_PASSWORD = settings.PG_PASSWORD

OLLAMA_BASE_URL    = settings.LLM_ENDPOINT
OLLAMA_EMBED_MODEL = settings.LLM_EMBEDDING_MODEL
OLLAMA_CHAT_MODEL  = settings.LLM_MODEL_NAME

MEMO_COLLECTION      = settings.MEMO_COLLECTION
MAX_HISTORY_MESSAGES = settings.MAX_HISTORY_MESSAGES
MAX_MSG_CHARS        = settings.MAX_MSG_CHARS

# ─── mem0 singleton ───────────────────────────────────────────────────────────

_memo_client: Optional[Memory] = None


def _build_memo_config() -> Dict[str, Any]:
    """
    Full mem0 config:
      - vector store : pgvector (same Postgres instance)
      - embedder     : Ollama
      - llm          : Ollama  (mem0 needs an LLM for memory distillation)
    """
    encoded_password = urllib.parse.quote_plus(PG_PASSWORD)
    return {
        "vector_store": {
            "provider": "pgvector",
            "config": {
                "host":       PG_HOST,
                "port":       PG_PORT,
                "dbname":     PG_DB,
                "user":       PG_USER,
                "password":   encoded_password,
                "collection_name": MEMO_COLLECTION,
                "embedding_model_dims": 2560,   # nomic-embed-text outputs 2560-d
            },
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model":    OLLAMA_EMBED_MODEL,
                "ollama_base_url": OLLAMA_BASE_URL,
            },
        },
        "llm": {
            "provider": "ollama",
            "config": {
                "model":    OLLAMA_CHAT_MODEL,
                "ollama_base_url": OLLAMA_BASE_URL,
                "temperature": 0.1,
            },
        },
    }


def get_memo_client() -> Memory:
    """Return (and lazily initialise) the global mem0 Memory client."""
    global _memo_client
    if _memo_client is None:
        # 1. Guarantee the halfvec table + index exist BEFORE mem0 touches them.
        conn = _get_pg_conn()          # also calls _ensure_schema internally
        _patch_mem0_for_halfvec(2560)  # must run before Memory.from_config

        logger.info("[Memo] Initialising mem0 with pgvector + Ollama backend …")
        _memo_client = Memory.from_config(_build_memo_config())
        logger.info("[Memo] mem0 client ready.")
    return _memo_client


def shutdown_memo() -> None:
    """Call on application shutdown to release resources cleanly."""
    global _memo_client
    if _memo_client is not None:
        try:
            _memo_client = None
            logger.info("[Memo] mem0 client released.")
        except Exception as exc:
            logger.warning("[Memo] Error during shutdown: %s", exc)


# ─── Raw PostgreSQL message log ───────────────────────────────────────────────
#
# mem0's pgvector store is great for *semantic* retrieval, but to replay
# an exact, ordered conversation we also keep a lightweight messages table.

_pg_conn: Optional[psycopg2.extensions.connection] = None


def _get_pg_conn() -> psycopg2.extensions.connection:
    """Return a persistent (reconnecting) psycopg2 connection."""
    global _pg_conn
    if _pg_conn is None or _pg_conn.closed:
        _pg_conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            dbname=PG_DB,
            user=PG_USER,
            password=PG_PASSWORD,
            cursor_factory=psycopg2.extras.DictCursor,
        )
        _pg_conn.autocommit = False
        _ensure_schema(_pg_conn)
    return _pg_conn

# ─── mem0 halfvec patch (must run before Memory.from_config) ──────────────────
def _patch_mem0_for_halfvec(dims: int = 2560) -> None:
    """
    mem0 hard-codes `vector(N)` + `vector_cosine_ops` in its DDL, which
    pgvector's HNSW index caps at 2000 dims.

    Instead of patching create_col (unreliable due to psycopg3 cursor
    context quirks), we patch psycopg3's Cursor.execute directly so every
    SQL string emitted by mem0 has `vector({dims})` → `halfvec({dims})`
    rewritten before it hits the server.  The rewrite is a no-op for any
    query that doesn't contain the exact token, so it won't affect other
    connections or unrelated queries.
    """
    try:
        import psycopg                         # psycopg3 — used by mem0
        from psycopg import Cursor, AsyncCursor

        _REPLACEMENTS = [
                    # CREATE TABLE column type
            (f"vector({dims})",   f"halfvec({dims})"),
            # CREATE INDEX operator classes
            ("vector_cosine_ops", "halfvec_cosine_ops"),
            ("vector_l2_ops",     "halfvec_l2_ops"),
            ("vector_ip_ops",     "halfvec_ip_ops"),
            # Search query cast  ($1::vector → $1::halfvec)
            ("::vector",          "::halfvec"),
        ]

        def _rewrite(sql: Any) -> Any:
            if not isinstance(sql, str):
                return sql
            for old, new in _REPLACEMENTS:
                sql = sql.replace(old, new)
            return sql

        # ── sync cursor ──
        _orig_execute = Cursor.execute

        def _patched_execute(self, query, params=None, *args, **kwargs):
            return _orig_execute(self, _rewrite(query), params, *args, **kwargs)

        Cursor.execute = _patched_execute

        # ── async cursor (mem0 may use either) ──
        _orig_aexecute = AsyncCursor.execute

        async def _patched_aexecute(self, query, params=None, *args, **kwargs):
            return await _orig_aexecute(self, _rewrite(query), params, *args, **kwargs)

        AsyncCursor.execute = _patched_aexecute

        logger.info(
            "[Memo] psycopg3 Cursor.execute patched: vector(%d) → halfvec(%d).", dims, dims
        )

    except Exception as exc:
        logger.warning("[Memo] Could not patch psycopg3 Cursor: %s", exc)

def _ensure_schema(conn: psycopg2.extensions.connection) -> None:
    """Create tables if they don't exist yet."""
    with conn.cursor() as cur:
        # pgvector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # DO NOT create MEMO_COLLECTION here — mem0 owns that table.
        # We only patch its DDL to use halfvec via _patch_mem0_for_halfvec().

        # Message log
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id          BIGSERIAL PRIMARY KEY,
                session_id  TEXT        NOT NULL,
                role        TEXT        NOT NULL,
                content     TEXT        NOT NULL,
                metadata    JSONB       DEFAULT '{}',
                created_at  TIMESTAMPTZ DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_chat_messages_session
                ON chat_messages (session_id, created_at);
            """
        )

        # Uploaded-document store
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS session_documents (
                session_id    TEXT        PRIMARY KEY,
                doc_text      TEXT        NOT NULL,
                filename      TEXT,
                char_count    INT,
                uploaded_at   TIMESTAMPTZ DEFAULT NOW()
            );
            """
        )
    conn.commit()
    logger.debug("[Memo] DB schema verified / created.")


# ─── Uploaded-document helpers ────────────────────────────────────────────────

def store_uploaded_document(
    session_id: str,
    doc_text: str,
    filename: str = "",
) -> None:
    """
    Persist the extracted text of an uploaded document for *session_id*.
    Replaces any previous upload for that session.

    Also adds a semantic memory entry to mem0 so the LLM can recall
    document facts across turns even after the raw text is dropped.
    """
    conn = _get_pg_conn()
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO session_documents (session_id, doc_text, filename, char_count)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (session_id) DO UPDATE
                SET doc_text    = EXCLUDED.doc_text,
                    filename    = EXCLUDED.filename,
                    char_count  = EXCLUDED.char_count,
                    uploaded_at = NOW()
            """,
            (session_id, doc_text, filename, len(doc_text)),
        )
    conn.commit()
    logger.debug("[Memo] Stored %d chars for session %s (%s)", len(doc_text), session_id, filename)

    # Add a distilled memory so future searches can surface it
    try:
        memo = get_memo_client()
        snippet = doc_text[:2000]           # mem0 distils this into key facts
        memo.add(
            [{"role": "user", "content": f"[UPLOADED DOCUMENT: {filename}]\n{snippet}"}],
            user_id=session_id,
            metadata={"type": "uploaded_doc", "filename": filename},
        )
    except Exception as exc:
        logger.warning("[Memo] mem0.add for uploaded doc failed: %s", exc)


def get_uploaded_document(session_id: str) -> Optional[str]:
    """Return the raw doc text stored for *session_id*, or None."""
    conn = _get_pg_conn()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT doc_text FROM session_documents WHERE session_id = %s",
            (session_id,),
        )
        row = cur.fetchone()
    return row["doc_text"] if row else None


def clear_uploaded_document(session_id: str) -> None:
    """Remove the uploaded document for *session_id*."""
    conn = _get_pg_conn()
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM session_documents WHERE session_id = %s",
            (session_id,),
        )
    conn.commit()


# ─── Core History Class ───────────────────────────────────────────────────────

class MemoChatMessageHistory(BaseChatMessageHistory):
    """
    LangChain-compatible chat history backed by PostgreSQL + mem0/pgvector.

    Drop-in replacement for FileChatMessageHistory.

    Attributes
    ----------
    session_id : str
        Unique identifier for the conversation (e.g. Socket.IO sid).
    use_semantic_context : bool
        When True, relevant past memories from mem0 are prepended as a
        SystemMessage so the LLM gets richer context without ballooning
        the token count with raw history.
    """

    def __init__(
        self,
        session_id: str,
        use_semantic_context: bool = True,
    ) -> None:
        self.session_id = session_id
        self.use_semantic_context = use_semantic_context

    # ── BaseChatMessageHistory interface ─────────────────────────────────────

    @property
    def messages(self) -> List[BaseMessage]:
        """
        Return the last MAX_HISTORY_MESSAGES ordered messages for this session,
        optionally prefixed with a SystemMessage containing semantic memories.
        """
        raw = self._load_raw_messages()

        if self.use_semantic_context and raw:
            # Build a short query from the last human turn for memory search
            last_human = next(
                (m.content for m in reversed(raw) if isinstance(m, HumanMessage)),
                "",
            )
            semantic_block = self._fetch_semantic_context(last_human)
            if semantic_block:
                return [SystemMessage(content=semantic_block)] + raw
        return raw

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Persist new messages to Postgres and index them in mem0."""
        conn = _get_pg_conn()
        with conn.cursor() as cur:
            for msg in messages:
                role = _role_from_message(msg)
                content = (msg.content or "").strip()
                cur.execute(
                    """
                    INSERT INTO chat_messages (session_id, role, content, metadata)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (
                        self.session_id,
                        role,
                        content,
                        json.dumps(getattr(msg, "additional_kwargs", {})),
                    ),
                )
        conn.commit()

        # Index in mem0 for semantic recall
        self._index_in_memo(messages)

    def clear(self) -> None:
        """Delete all messages and memories for this session."""
        conn = _get_pg_conn()
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM chat_messages WHERE session_id = %s",
                (self.session_id,),
            )
        conn.commit()

        # Delete mem0 memories for this user
        try:
            memo = get_memo_client()
            memo.delete_all(user_id=self.session_id)
        except Exception as exc:
            logger.warning("[Memo] mem0 delete_all failed for %s: %s", self.session_id, exc)

        clear_uploaded_document(self.session_id)
        logger.info("[Memo] Cleared all history for session %s", self.session_id)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_raw_messages(self) -> List[BaseMessage]:
        """
        Fetch the last MAX_HISTORY_MESSAGES from Postgres,
        truncate each to MAX_MSG_CHARS chars.
        """
        conn = _get_pg_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT role, content, metadata
                FROM   chat_messages
                WHERE  session_id = %s
                ORDER  BY created_at DESC, id DESC  -- Add id DESC for stability
                LIMIT  %s
                """,
                (self.session_id, MAX_HISTORY_MESSAGES),
            )
            rows = cur.fetchall()

        # Reverse so oldest-first
        rows = list(reversed(rows))

        result: List[BaseMessage] = []
        for row in rows:
            content = (row["content"] or "").strip()[:MAX_MSG_CHARS]
            role    = row["role"]
            metadata = row["metadata"] or {}
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}
            if role == "human":
                result.append(HumanMessage(content=content, additional_kwargs=metadata))
            elif role == "ai":
                result.append(AIMessage(content=content, additional_kwargs=metadata))
            elif role == "system":
                result.append(SystemMessage(content=content, additional_kwargs=metadata))
        return result

    def _fetch_semantic_context(self, query: str) -> str:
        """
        Search mem0 for relevant past memories and return them as a
        formatted string to prepend as a SystemMessage.
        Returns "" if nothing relevant is found or on any error.
        """
        if not query.strip():
            return ""
        try:
            memo   = get_memo_client()
            results = memo.search(query, user_id=self.session_id, limit=5)

            # mem0 returns {"results": [{"memory": "...", "score": float}, ...]}
            memories = results.get("results", []) if isinstance(results, dict) else results

            if not memories:
                return ""

            lines = ["[Relevant past context from this conversation:]"]
            for item in memories:
                mem_text = item.get("memory") or item.get("text") or str(item)
                score    = item.get("score", 0.0)
                if score > 0.40:           # only include sufficiently relevant hits
                    lines.append(f"• {mem_text.strip()}")

            return "\n".join(lines) if len(lines) > 1 else ""
        except Exception as exc:
            logger.debug("[Memo] Semantic context fetch failed: %s", exc)
            return ""

    def _index_in_memo(self, messages: Sequence[BaseMessage]) -> None:
        """Send new messages to mem0 for distillation + vector indexing."""
        try:
            memo = get_memo_client()
            payload = [
                {"role": _role_from_message(m), "content": (m.content or "").strip()}
                for m in messages
                if (m.content or "").strip()
            ]
            if payload:
                memo.add(payload, user_id=self.session_id)
        except Exception as exc:
            logger.warning("[Memo] mem0.add failed: %s", exc)

    # ── Convenience: async wrappers ───────────────────────────────────────────

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Async-safe wrapper — offloads blocking I/O to a thread."""
        await asyncio.to_thread(self.add_messages, messages)

    async def aclear(self) -> None:
        await asyncio.to_thread(self.clear)

    # ── Uploaded document shortcut ────────────────────────────────────────────

    def save_document(self, doc_text: str, filename: str = "") -> None:
        """Store an uploaded document and add it to mem0 semantic index."""
        store_uploaded_document(self.session_id, doc_text, filename)

    def load_document(self) -> Optional[str]:
        """Retrieve the uploaded document text for this session."""
        return get_uploaded_document(self.session_id)

    def __repr__(self) -> str:
        return f"MemoChatMessageHistory(session_id={self.session_id!r})"


# ─── Factory / Store ──────────────────────────────────────────────────────────

# In-memory cache so we don't reconstruct the object on every call
_history_store: Dict[str, MemoChatMessageHistory] = {}


def get_session_history(session_id: str) -> MemoChatMessageHistory:
    """
    Return (and cache) a MemoChatMessageHistory for *session_id*.

    Usage in server.py — replace:
        from server import get_session_history
    with:
        from context_memory import get_session_history
    """
    if session_id not in _history_store:
        _history_store[session_id] = MemoChatMessageHistory(session_id=session_id)
    return _history_store[session_id]


def evict_session(session_id: str) -> None:
    """Remove a session from the in-memory cache (call on disconnect)."""
    _history_store.pop(session_id, None)


# ─── Utility ──────────────────────────────────────────────────────────────────

def _role_from_message(msg: BaseMessage) -> str:
    """Map a LangChain message type to a plain role string."""
    if isinstance(msg, HumanMessage):
        return "human"
    if isinstance(msg, AIMessage):
        return "ai"
    if isinstance(msg, SystemMessage):
        return "system"
    # Fallback: try the `type` attribute that all messages carry
    return getattr(msg, "type", "human")