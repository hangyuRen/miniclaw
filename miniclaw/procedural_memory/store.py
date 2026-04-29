"""SQLite + FTS5 + vec0 (sqlite-vec) storage for procedural memory with hybrid search."""

from __future__ import annotations

import json
import math
import sqlite3
import struct
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from loguru import logger


@dataclass
class Procedure:
    id: int
    user_id: str
    title: str
    description: str
    steps: str
    tags: list[str]
    usage_count: int
    created_at: str
    updated_at: str


# ---------------------------------------------------------------------------
# Schema — base tables (no vec0 dependency)
# ---------------------------------------------------------------------------

_DDL_BASE = """
CREATE TABLE IF NOT EXISTS procedures (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL DEFAULT 'shared',
    title       TEXT NOT NULL,
    description TEXT NOT NULL,
    steps       TEXT NOT NULL,
    tags        TEXT NOT NULL DEFAULT '[]',
    usage_count INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_procedures_user_id ON procedures(user_id);

-- embedding_text: source text used to generate the vector
-- embedding_blob: struct-packed float32 fallback (used when vec0 unavailable)
CREATE TABLE IF NOT EXISTS procedures_vec (
    proc_id        INTEGER PRIMARY KEY REFERENCES procedures(id) ON DELETE CASCADE,
    embedding_text TEXT NOT NULL,
    embedding_blob BLOB
);

CREATE VIRTUAL TABLE IF NOT EXISTS procedures_fts USING fts5(
    title,
    description,
    steps,
    tags,
    content=procedures,
    content_rowid=id,
    tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS procedures_ai AFTER INSERT ON procedures BEGIN
    INSERT INTO procedures_fts(rowid, title, description, steps, tags)
    VALUES (new.id, new.title, new.description, new.steps, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS procedures_au AFTER UPDATE ON procedures BEGIN
    INSERT INTO procedures_fts(procedures_fts, rowid, title, description, steps, tags)
    VALUES ('delete', old.id, old.title, old.description, old.steps, old.tags);
    INSERT INTO procedures_fts(rowid, title, description, steps, tags)
    VALUES (new.id, new.title, new.description, new.steps, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS procedures_ad AFTER DELETE ON procedures BEGIN
    INSERT INTO procedures_fts(procedures_fts, rowid, title, description, steps, tags)
    VALUES ('delete', old.id, old.title, old.description, old.steps, old.tags);
END;
"""

# vec0 KNN table — created only when sqlite-vec extension loads successfully.
# Dimension is substituted at runtime from config.
_DDL_VEC0 = """
CREATE VIRTUAL TABLE IF NOT EXISTS procedures_vec_knn USING vec0(
    proc_id INTEGER PRIMARY KEY,
    embedding float[{dim}]
);
"""

_MIGRATIONS: list[tuple[str, str]] = [
    ("user_id", "ALTER TABLE procedures ADD COLUMN user_id TEXT NOT NULL DEFAULT 'shared'"),
]
_MIGRATION_IDX = "CREATE INDEX IF NOT EXISTS idx_procedures_user_id ON procedures(user_id)"
_MIGRATION_VEC_TABLE = """
CREATE TABLE IF NOT EXISTS procedures_vec (
    proc_id        INTEGER PRIMARY KEY REFERENCES procedures(id) ON DELETE CASCADE,
    embedding_text TEXT NOT NULL,
    embedding_blob BLOB
)
"""


# ---------------------------------------------------------------------------
# Pure-Python cosine (fallback)
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _pack_vec(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack_vec(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class ProceduralMemoryStore:
    """
    SQLite-backed store for procedural memory.

    Vector search strategy (in order of preference):
      1. sqlite-vec KNN via vec0 virtual table (fast, in-DB)
      2. Python cosine over stored embedding_blob (fallback when vec0 unavailable)

    The vec0 extension (vec0.dll / vec0.so) is expected next to the database file.
    If loading fails, the store silently falls back to Python-side similarity.
    """

    def __init__(self, db_path: str | Path, embed_dim: int = 1536):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embed_dim = embed_dim
        # vec0.dll / vec0.so lives next to the database file (same directory)
        self._vec_ext_path = str(self.db_path.parent / "vec0")
        self._vec0_available: bool | None = None  # None = not yet probed
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    # Connection & extension loading
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._try_load_vec0(self._conn)
            self._conn.executescript(_DDL_BASE)
            self._migrate(self._conn)
            if self._vec0_available:
                self._ensure_vec0_table(self._conn)
            self._conn.commit()
        return self._conn

    def _try_load_vec0(self, conn: sqlite3.Connection) -> None:
        """Attempt to load the sqlite-vec extension. Sets self._vec0_available."""
        if self._vec0_available is not None:
            return
        try:
            conn.enable_load_extension(True)
            conn.execute("SELECT load_extension(?)", (self._vec_ext_path,))
            version = conn.execute("SELECT vec_version()").fetchone()[0]
            logger.info("sqlite-vec loaded ({}); KNN vector search enabled.", version)
            self._vec0_available = True
        except Exception as e:
            logger.debug("sqlite-vec unavailable ({}): falling back to Python cosine.", e)
            self._vec0_available = False

    def _ensure_vec0_table(self, conn: sqlite3.Connection) -> None:
        conn.executescript(_DDL_VEC0.format(dim=self.embed_dim))

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        existing = {row[1] for row in conn.execute("PRAGMA table_info(procedures)")}
        for col, sql in _MIGRATIONS:
            if col not in existing:
                conn.execute(sql)
        conn.execute(_MIGRATION_IDX)
        conn.execute(_MIGRATION_VEC_TABLE)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def vec0_available(self) -> bool:
        """True after _get_conn() has been called and extension loaded successfully."""
        return bool(self._vec0_available)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def insert(self, title: str, description: str, steps: str, tags: list[str], user_id: str = "shared") -> int:
        now = datetime.utcnow().isoformat()
        conn = self._get_conn()
        cur = conn.execute(
            "INSERT INTO procedures (user_id, title, description, steps, tags, usage_count, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, 1, ?, ?)",
            (user_id, title, description, steps, json.dumps(tags, ensure_ascii=False), now, now),
        )
        proc_id: int = cur.lastrowid  # type: ignore[assignment]
        embedding_text = self._make_embedding_text(title, description, steps, tags)
        conn.execute(
            "INSERT INTO procedures_vec (proc_id, embedding_text) VALUES (?, ?)",
            (proc_id, embedding_text),
        )
        conn.commit()
        return proc_id

    def update(self, proc_id: int, title: str, description: str, steps: str, tags: list[str]) -> None:
        now = datetime.utcnow().isoformat()
        conn = self._get_conn()
        conn.execute(
            "UPDATE procedures SET title=?, description=?, steps=?, tags=?, "
            "usage_count=usage_count+1, updated_at=? WHERE id=?",
            (title, description, steps, json.dumps(tags, ensure_ascii=False), now, proc_id),
        )
        embedding_text = self._make_embedding_text(title, description, steps, tags)
        # Reset embedding so it gets re-computed on next retrieve
        conn.execute(
            "INSERT INTO procedures_vec (proc_id, embedding_text, embedding_blob) VALUES (?, ?, NULL) "
            "ON CONFLICT(proc_id) DO UPDATE SET embedding_text=excluded.embedding_text, embedding_blob=NULL",
            (proc_id, embedding_text),
        )
        # Remove stale vector from vec0 index so it's not returned by KNN
        if self._vec0_available:
            conn.execute("DELETE FROM procedures_vec_knn WHERE proc_id=?", (proc_id,))
        conn.commit()

    def store_embedding(self, proc_id: int, vec: list[float]) -> None:
        """Persist a computed embedding — both the blob fallback and the vec0 index."""
        blob = _pack_vec(vec)
        conn = self._get_conn()
        # Always store blob (used as fallback and for inspection)
        conn.execute(
            "UPDATE procedures_vec SET embedding_blob=? WHERE proc_id=?",
            (blob, proc_id),
        )
        # vec0 does not support ON CONFLICT; delete then insert
        if self._vec0_available:
            conn.execute("DELETE FROM procedures_vec_knn WHERE proc_id=?", (proc_id,))
            conn.execute(
                "INSERT INTO procedures_vec_knn(proc_id, embedding) VALUES (?, ?)",
                (proc_id, blob),
            )
        conn.commit()

    def bump_usage(self, proc_id: int) -> None:
        now = datetime.utcnow().isoformat()
        conn = self._get_conn()
        conn.execute(
            "UPDATE procedures SET usage_count=usage_count+1, updated_at=? WHERE id=?",
            (now, proc_id),
        )
        conn.commit()

    def delete(self, proc_id: int) -> bool:
        conn = self._get_conn()
        if self._vec0_available:
            conn.execute("DELETE FROM procedures_vec_knn WHERE proc_id=?", (proc_id,))
        cur = conn.execute("DELETE FROM procedures WHERE id=?", (proc_id,))
        conn.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------------
    # BM25 search (FTS5, always available)
    # ------------------------------------------------------------------

    def bm25_search(self, query: str, user_id: str = "shared", top_k: int = 10) -> list[tuple[int, float]]:
        """Return [(proc_id, bm25_score)] — lower score = more relevant."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT p.id, bm25(procedures_fts) AS score
                FROM procedures_fts f
                JOIN procedures p ON p.id = f.rowid
                WHERE procedures_fts MATCH ?
                  AND p.user_id = ?
                ORDER BY score ASC
                LIMIT ?
                """,
                (query, user_id, top_k),
            ).fetchall()
            return [(int(r["id"]), float(r["score"])) for r in rows]
        except sqlite3.OperationalError as e:
            logger.warning("BM25 search failed: {}", e)
            return []

    # ------------------------------------------------------------------
    # Vector search — vec0 KNN (preferred) or Python cosine (fallback)
    # ------------------------------------------------------------------

    def vector_search(self, query_vec: list[float], user_id: str = "shared", top_k: int = 10) -> list[tuple[int, float]]:
        """
        Return [(proc_id, score)] sorted by relevance descending.

        Uses vec0 KNN when available (score = 1 - L2_distance, approximated),
        otherwise falls back to Python cosine over stored blobs.
        """
        conn = self._get_conn()
        if self._vec0_available:
            return self._vec0_knn(conn, query_vec, user_id, top_k)
        return self._python_cosine(conn, query_vec, user_id, top_k)

    def _vec0_knn(
        self, conn: sqlite3.Connection, query_vec: list[float], user_id: str, top_k: int
    ) -> list[tuple[int, float]]:
        """KNN via vec0 virtual table, pre-filtered by user_id subquery."""
        blob = _pack_vec(query_vec)
        try:
            rows = conn.execute(
                """
                SELECT v.proc_id, v.distance
                FROM procedures_vec_knn v
                WHERE v.embedding MATCH ?
                  AND v.proc_id IN (
                      SELECT id FROM procedures WHERE user_id = ?
                  )
                ORDER BY v.distance ASC
                LIMIT ?
                """,
                (blob, user_id, top_k),
            ).fetchall()
            # vec0 returns L2 distance (lower = closer); invert to score
            return [(int(r["proc_id"]), 1.0 / (1.0 + float(r["distance"]))) for r in rows]
        except Exception as e:
            logger.warning("vec0 KNN failed, falling back to Python cosine: {}", e)
            return self._python_cosine(conn, query_vec, user_id, top_k)

    def _python_cosine(
        self, conn: sqlite3.Connection, query_vec: list[float], user_id: str, top_k: int
    ) -> list[tuple[int, float]]:
        """Cosine similarity computed in Python over all stored blobs for the user."""
        rows = conn.execute(
            """
            SELECT v.proc_id, v.embedding_blob
            FROM procedures_vec v
            JOIN procedures p ON p.id = v.proc_id
            WHERE p.user_id = ?
              AND v.embedding_blob IS NOT NULL
            """,
            (user_id,),
        ).fetchall()
        if not rows:
            return []
        scored = [
            (int(r["proc_id"]), _cosine_similarity(query_vec, _unpack_vec(bytes(r["embedding_blob"]))))
            for r in rows
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Pending embeddings
    # ------------------------------------------------------------------

    def get_pending_embedding(self, user_id: str | None = None) -> list[tuple[int, str]]:
        """Return [(proc_id, embedding_text)] for rows without a stored embedding."""
        conn = self._get_conn()
        if user_id:
            rows = conn.execute(
                """
                SELECT v.proc_id, v.embedding_text
                FROM procedures_vec v
                JOIN procedures p ON p.id = v.proc_id
                WHERE v.embedding_blob IS NULL AND p.user_id = ?
                """,
                (user_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT proc_id, embedding_text FROM procedures_vec WHERE embedding_blob IS NULL"
            ).fetchall()
        return [(int(r["proc_id"]), r["embedding_text"]) for r in rows]

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_by_id(self, proc_id: int) -> Procedure | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT id, user_id, title, description, steps, tags, usage_count, created_at, updated_at "
            "FROM procedures WHERE id=?",
            (proc_id,),
        ).fetchone()
        return self._row_to_proc(row) if row else None

    def get_by_ids(self, proc_ids: list[int]) -> dict[int, Procedure]:
        if not proc_ids:
            return {}
        placeholders = ",".join("?" * len(proc_ids))
        conn = self._get_conn()
        rows = conn.execute(
            f"SELECT id, user_id, title, description, steps, tags, usage_count, created_at, updated_at "
            f"FROM procedures WHERE id IN ({placeholders})",
            proc_ids,
        ).fetchall()
        return {int(r["id"]): self._row_to_proc(r) for r in rows}

    def list_all(self, user_id: str | None = None) -> list[Procedure]:
        conn = self._get_conn()
        if user_id is not None:
            rows = conn.execute(
                "SELECT id, user_id, title, description, steps, tags, usage_count, created_at, updated_at "
                "FROM procedures WHERE user_id=? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, user_id, title, description, steps, tags, usage_count, created_at, updated_at "
                "FROM procedures ORDER BY updated_at DESC"
            ).fetchall()
        return [self._row_to_proc(r) for r in rows]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_embedding_text(title: str, description: str, steps: str, tags: list[str]) -> str:
        tag_str = " ".join(tags)
        return f"{title}\n{description}\n{steps}\n{tag_str}".strip()

    @staticmethod
    def _row_to_proc(row: sqlite3.Row) -> Procedure:
        tags = json.loads(row["tags"]) if row["tags"] else []
        return Procedure(
            id=row["id"],
            user_id=row["user_id"],
            title=row["title"],
            description=row["description"],
            steps=row["steps"],
            tags=tags,
            usage_count=row["usage_count"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
