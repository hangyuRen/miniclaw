"""Personal long-term memory store backed by SQLite."""

from __future__ import annotations

import json
import math
import sqlite3
import uuid
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from miniclaw.config.schema import MemorySystemConfig
from miniclaw.utils.helpers import ensure_dir


KIND_WEIGHTS = {
    "constraint": 1.4,
    "preference": 1.3,
    "decision": 1.2,
    "reference": 1.0,
    "profile": 0.8,
}


class PersonalMemoryStore:
    """SQLite-backed canonical personal memory store."""

    def __init__(self, workspace: Path, config: MemorySystemConfig):
        self.workspace = workspace
        self.config = config
        self.db_path = Path(config.db_path).expanduser()
        self.sqlite_vec_path = workspace / "memory" / "vec0.dll"
        ensure_dir(self.db_path.parent)
        self.memory_md_path = workspace / "memory" / "MEMORY.md"
        self.init_db()

    def init_db(self) -> None:
        with self._connect() as conn:
             # 长期记忆存储表
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_chunks (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    status TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                    )
                """
            )

            # 在embedding上建立索引 加速查询
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_memory_chunks USING vec0(
                    chunk_id TEXT PRIMARY KEY, 
                    embedding float[384]
                )
                """
            )

            # 建立FTS5虚拟表
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                    content,
                    chunk_id UNINDEXED,
                    tokenize = "trigram"
                )
                """
            )

            # 记忆候选表：存放刚从对话中提取出来的原始记忆
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_candidates (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    extracted_from TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    scope_key TEXT DEFAULT '',
                    slot TEXT DEFAULT '',
                    content TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    keywords_json TEXT NOT NULL,
                    source_refs_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    merged INTEGER NOT NULL DEFAULT 0
                )
                """
            )

            # 记忆事件表：记录记忆的演进过程
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_events (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    memory_id TEXT,
                    candidate_id TEXT,
                    action TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    before_json TEXT,
                    after_json TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )

            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        # 返回的数据是字典格式 但是不能修改
        conn.row_factory = sqlite3.Row
        # 尝试加载sqlite-vec扩展
        try:
            conn.enable_load_extension(True)
            # 根据操作系统加载不同的扩展文件
            conn.execute("SELECT load_extension(?)", (str(self.sqlite_vec_path),))  # Windows系统
            # 测试扩展是否成功加载
            cursor = conn.cursor()
            cursor.execute("SELECT vec_version()")
            version = cursor.fetchone()[0]
            print(f"成功加载sqlite-vec扩展, 版本: {version}")
        except Exception as e:
            print(f"加载sqlite-vec扩展失败: {e}")
            print("将使用纯Python实现向量操作")
            print("如需使用sqlite-vec扩展, 请从 https://github.com/asg017/sqlite-vec/releases 下载对应版本")
        return conn

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).astimezone().isoformat()

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _loads_json(value: str | None, default: Any) -> Any:
        if not value:
            return default
        try:
            return json.loads(value)
        except Exception:
            return default
        
    # SQLite中存储的是字符串格式 取出来需要转化为json格式 方便后续操作
    def _row_to_memory(self, row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        data = dict(row)
        return data


    def _upsert_fts(self, conn: sqlite3.Connection, chunk_id: str, content: str) -> None:
        conn.execute("DELETE FROM memory_fts WHERE chunk_id = ?", (chunk_id,))
        conn.execute(
            "INSERT INTO memory_fts(chunk_id, content) VALUES (?, ?)",
            (chunk_id, content),
        )
    

    def add_candidates(self, candidates: list[dict[str, Any]], extracted_from: str, user_id: str | None = None) -> list[str]:
        if not candidates:
            return []
        user_id = user_id or self.config.default_user_id
        created_ids: list[str] = []
        with self._connect() as conn:
            for candidate in candidates[: max(1, self.config.max_candidates_per_run)]:
                cid = candidate.get("id") or f"cand_{uuid.uuid4().hex[:12]}"
                created_ids.append(cid)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO memory_candidates(
                        id, user_id, extracted_from, kind, scope, scope_key, slot,
                        content, summary, tags_json, keywords_json, source_refs_json,
                        created_at, merged
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                    """,
                    (
                        cid,
                        user_id,
                        extracted_from,
                        candidate.get("kind", "reference"),
                        candidate.get("scope", "global"),
                        candidate.get("scope_key", ""),
                        candidate.get("slot", ""),
                        candidate.get("content", "").strip(),
                        candidate.get("summary", "").strip(),
                        self._json(candidate.get("tags", [])),
                        self._json(candidate.get("keywords", [])),
                        self._json(candidate.get("source_refs", [])),
                        self._now_iso(),
                    ),
                )
            conn.commit()
        return created_ids

    def get_unmerged_candidates(self, limit: int = 100, user_id: str | None = None) -> list[dict[str, Any]]:
        user_id = user_id or self.config.default_user_id
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memory_candidates WHERE user_id = ? AND merged = 0 ORDER BY created_at ASC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        result = []
        for row in rows:
            data = dict(row)
            for key in ("tags_json", "keywords_json", "source_refs_json"):
                parsed_key = key.replace("_json", "")
                data[parsed_key] = self._loads_json(data.pop(key, "[]"), [])
            result.append(data)
        return result


    def get_memory(self, chunk_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM memory_chunks WHERE id = ?", (chunk_id,)).fetchone()
        return self._row_to_memory(row)
    

    def _upsert_vec(self, conn: sqlite3.Connection, chunk_id: str, content: str) -> None:
        conn.execute("DELETE FROM vec_memory_chunks WHERE chunk_id = ?", (chunk_id,))
        conn.execute(
            "INSERT INTO vec_memory_chunks(chunk_id, embedding) VALUES (?, ?)",
            (chunk_id, content),
        )

    
    def create_memory(self, memory: dict[str, Any], user_id: str | None = None) -> str:
        user_id = user_id or self.config.default_user_id
        chunk_id = memory.get("id") or f"mem_{uuid.uuid4().hex[:12]}"
        now = self._now_iso()
        embedding_data = memory.get("embedding")
        embedding = struct.pack(f"{len(embedding_data)}f", *embedding_data)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_chunks(
                    id, user_id, content, status, updated_at, created_at
                ) VALUES (?, ?, ?, 'active', ?, ?)
                """,
                (
                    chunk_id,
                    user_id,
                    memory.get("slot", ""),
                    memory.get("content", "").strip(),
                    embedding,
                    now,
                    now,
                ),
            )
            self._upsert_vec(conn, chunk_id, memory.get("content", ""))
            self._upsert_fts(conn, chunk_id, memory.get("content", ""))
            conn.commit()
        return chunk_id

    
    def update_memory(self, chunk_id: str, new_memory: dict[str, Any]) -> None:
        current = self.get_memory(chunk_id)
        if not current:
            raise ValueError(f"Memory not found: {chunk_id}")
        with self._connect() as conn:
            embedding_data = new_memory.get("embedding")
            embedding = struct.pack(f"{len(embedding_data)}f", *embedding_data)
            conn.execute(
                """
                UPDATE memory_chunks SET
                    content = ?,
                    status = 'active',
                    updated_at = ?,
                WHERE id = ?
                """,
                (
                    new_memory.get("content", current.get("content", "")).strip(),
                    self._now_iso(),
                    chunk_id,
                ),
            )
            self._upsert_vec(
                conn,
                chunk_id,
                new_memory.get("content", current.get("content", ""))
            )
            self._upsert_fts(
                conn,
                chunk_id,
                new_memory.get("content", current.get("content", ""))
            )
            conn.commit()

    
    def supersede_memory(self, old_id: str, new_memory: dict[str, Any], user_id: str | None = None) -> str:
        user_id = user_id or self.config.default_user_id
        with self._connect() as conn:
            conn.execute("UPDATE memory_chunks SET status = 'superseded', updated_at = ? WHERE id = ?", (self._now_iso(), old_id))
            conn.execute("DELETE FROM vec_memory_chunks WHERE chunk_id = ?", (old_id,))
            conn.execute("DELETE FROM memory_fts WHERE chunk_id = ?", (old_id,))
            conn.commit()
        return self.create_memory(new_memory, user_id=user_id)

    
    def archive_memory(self, chunk_id: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE memory_chunks SET status = 'archived', updated_at = ? WHERE id = ?", (self._now_iso(), chunk_id))
            conn.execute("DELETE FROM vec_memory_chunks WHERE chunk_id = ?", (chunk_id,))
            conn.execute("DELETE FROM memory_fts WHERE chunk_id = ?", (chunk_id,))
            conn.commit()

    
    def mark_candidate_merged(self, candidate_id: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE memory_candidates SET merged = 1 WHERE id = ?", (candidate_id,))
            conn.commit()

    
    def log_event(self, event: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_events(
                    id, user_id, memory_id, candidate_id, action, reason,
                    before_json, after_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.get("id") or f"evt_{uuid.uuid4().hex[:12]}",
                    event.get("user_id") or self.config.default_user_id,
                    event.get("memory_id"),
                    event.get("candidate_id"),
                    event.get("action", "noop"),
                    event.get("reason", ""),
                    self._json(event.get("before")) if event.get("before") is not None else None,
                    self._json(event.get("after")) if event.get("after") is not None else None,
                    event.get("created_at") or self._now_iso(),
                ),
            )
            conn.commit()


    def list_active_memories(self, user_id: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
        user_id = user_id or self.config.default_user_id
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memory_chunks WHERE user_id = ? AND status = 'active' ORDER BY updated_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [self._row_to_memory(row) for row in rows if row is not None]

    
    def get_stats(self, user_id: str | None = None) -> dict[str, Any]:
        user_id = user_id or self.config.default_user_id
        with self._connect() as conn:
            active = conn.execute(
                "SELECT COUNT(*) FROM memory_chunks WHERE user_id = ? AND status = 'active'",
                (user_id,),
            ).fetchone()[0]
            superseded = conn.execute(
                "SELECT COUNT(*) FROM memory_chunks WHERE user_id = ? AND status = 'superseded'",
                (user_id,),
            ).fetchone()[0]
            archived = conn.execute(
                "SELECT COUNT(*) FROM memory_chunks WHERE user_id = ? AND status = 'archived'",
                (user_id,),
            ).fetchone()[0]
            candidates_unmerged = conn.execute(
                "SELECT COUNT(*) FROM memory_candidates WHERE user_id = ? AND merged = 0",
                (user_id,),
            ).fetchone()[0]
            candidates_total = conn.execute(
                "SELECT COUNT(*) FROM memory_candidates WHERE user_id = ?",
                (user_id,),
            ).fetchone()[0]
            events_total = conn.execute(
                "SELECT COUNT(*) FROM memory_events WHERE user_id = ?",
                (user_id,),
            ).fetchone()[0]
            latest_update = conn.execute(
                "SELECT updated_at FROM memory_chunks WHERE user_id = ? AND status = 'active' ORDER BY updated_at DESC LIMIT 1",
                (user_id,),
            ).fetchone()
        return {
            "user_id": user_id,
            "active": int(active or 0),
            "superseded": int(superseded or 0),
            "archived": int(archived or 0),
            "candidates_unmerged": int(candidates_unmerged or 0),
            "candidates_total": int(candidates_total or 0),
            "events_total": int(events_total or 0),
            "latest_update": latest_update[0] if latest_update else None,
            "db_path": str(self.db_path),
        }

    
    def list_recent_events(self, user_id: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        user_id = user_id or self.config.default_user_id
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memory_events WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            data = dict(row)
            for key in ("before_json", "after_json"):
                parsed_key = key.replace("_json", "")
                data[parsed_key] = self._loads_json(data.pop(key, None), None)
            result.append(data)
        return result
    

    def text2embed(self, query: str) -> list[float]:
        pass


    async def vector_search(self, user_id: str, query: str, top_k: int = 5) -> list[tuple[str, dict]]:
        query_vec = self.text2embed(query)
        query_blob = struct.pack(f"{len(query_vec)}f", *query_vec)
        with self._connect() as conn:
            # 获取该用户所有有效的记忆块及其向量
            rows = conn.execute(
                """
                SELECT c.id, c.content, v.distance
                FROM vec_memory_chunks v
                JOIN memory_chunks c ON v.chunk_id = c.id
                WHERE v.embedding MATCH ? 
                AND c.user_id = ? 
                AND c.status = 'active'
                ORDER BY v.distance ASC
                LIMIT ?
                """,
                (user_id, query_blob, top_k)
            ).fetchall()
              
            return [(row["id"], dict(row)) for row in rows]

    
    async def bm25_search(self, user_id: str, query: str, top_k: int = 5) -> list[tuple[str, dict]]:
        tokens = self._tokenize(query)
        if not tokens:
            return []
        
        # 构造 FTS5 查询语法
        fts_query = " ".join([f'"{t}"' for t in tokens])
        
        with self._connect() as conn:
            # sqlite 的 bm25() 分值越小越相关
            rows = conn.execute(
                """
                SELECT chunk_id, bm25(memory_fts) as score
                FROM memory_fts f
                JOIN memory_chunks c ON f.chunk_id = c.id
                WHERE f.content MATCH ? AND c.user_id = ? AND c.status = 'active'
                ORDER BY score ASC
                LIMIT ?
                """,
                (fts_query, user_id, top_k),
            ).fetchall()
            
            return [(row["chunk_id"], dict(row)) for row in rows]

    
    async def hybrid_retrieve(self, user_id: str | None, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        # 1. 获取两路检索的原始结果
        vector_results = await self.vector_search(user_id, query, limit=top_k * 2)
        bm25_results = await self.bm25_search(user_id, query, limit=top_k * 2)

        # 2. RRF 重排逻辑
        k_rrf = 60
        rrf_scores: dict[str, float] = {}
        data_cache: dict[str, dict] = {}

        # 处理向量路结果: list of (memory_id, score)
        for rank, (m_id, data) in enumerate(vector_results, start=1):
            rrf_scores[m_id] = rrf_scores.get(m_id, 0.0) + 1.0 / (k_rrf + rank)
            data_cache[m_id] = data

        # 处理 BM25 路结果: list of (memory_id, score)
        for rank, (m_id, data) in enumerate(bm25_results, start=1):
            rrf_scores[m_id] = rrf_scores.get(m_id, 0.0) + 1.0 / (k_rrf + rank)
            data[m_id] = data

        # 3. 按 RRF 分数排序并取回完整对象
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return [data_cache[m_id] for m_id, _ in sorted_ids]
    

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        if not text:
            return []
        rough = []
        buf = []
        for ch in text:
            if ch.isalnum() or ch in {"_", "-", "/", "."}:
                buf.append(ch)
            else:
                if buf:
                    rough.append("".join(buf))
                    buf = []
                if not ch.isspace():
                    rough.append(ch)
        if buf:
            rough.append("".join(buf))
        tokens = [t for t in rough if t and t.strip()]
        return list(dict.fromkeys(tokens))[:64]

    
    def list_core_candidates(self, user_id: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        user_id = user_id or self.config.default_user_id
        limit = limit or self.config.core_memory_max_items
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM memory_chunks
                WHERE user_id = ? AND status = 'active'
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return [self._row_to_memory(row) for row in rows if row is not None]
    

    def sync_memory_md(self, extra_notes: list[str] | None = None, user_id: str | None = None) -> None:
        """Rewrite MEMORY.md index with integrated core memory section."""
        if not self.config.update_memory_md:
            return
        user_id = user_id or self.config.default_user_id
        existing = self.memory_md_path.read_text(encoding="utf-8") if self.memory_md_path.exists() else "# MEMORY\n"
        prefix, _, _ = existing.partition("## Auto Core Memory")
        prefix = prefix.rstrip()
        core_items = self.list_core_candidates(user_id=user_id, limit=self.config.core_memory_max_items)
        lines = []
        lines.append("## Auto Core Memory")
        lines.append("")
        if core_items:
            for item in core_items:
                lines.append(f"- {item.get('content') or ""}")
        else:
            lines.append("- (暂无自动核心记忆条目)")
        if extra_notes:
            lines.append("")
            lines.append("## Auto Memory System")
            lines.append("")
            for note in extra_notes:
                lines.append(f"- {note}")
        content = prefix + "\n\n" + "\n".join(lines) + "\n"
        self.memory_md_path.write_text(content, encoding="utf-8")


if __name__=='__main__':
    p = PersonalMemoryStore(Path('C:\\Users\\hyuren\\.miniclaw\\workspace'), MemorySystemConfig())