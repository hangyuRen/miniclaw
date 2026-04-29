"""Retrieve relevant personal memories for prompt injection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from miniclaw.agent.personal_memory_store_new import PersonalMemoryStore
from miniclaw.config.schema import MemorySystemConfig


class MemoryRetriever:
    """Build retrieval query and render prompt blocks for personal memory."""

    def __init__(self, workspace: Path, config: MemorySystemConfig, user_id: str):
        self.workspace = workspace
        self.config = config
        self.store = PersonalMemoryStore(workspace, config)
        self.user_id = user_id

    def build_query(
        self,
        user_text: str,
        session_state: str | None,
        recent_messages: list[dict[str, Any]],
    ) -> str:
        parts: list[str] = []
        if session_state:
            parts.append(str(session_state).strip())
        for msg in recent_messages[-4:]:
            role = msg.get("role")
            content = str(msg.get("content") or "").strip()
            if role in {"user", "assistant"} and content:
                parts.append(content)
        if user_text.strip():
            parts.append(user_text.strip())
        return "\n".join(parts).strip()

    
    async def retrieve_for_prompt(
        self,
        user_text: str,
        session_state: str | None,
        recent_messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        query = self.build_query(user_text, session_state, recent_messages)
        results = await self.store.hybrid_retrieve(user_id=self.user_id, query=query, top_k=self.config.retrieval_top_k)
        return results

    
    def render_memory_block(self, retrieved: list[dict[str, Any]]) -> str:
        if not retrieved:
            return ""
        lines = ["## Retrieved Personal Memory", ""]
        for item in retrieved:
            content = item.get("content") or ""
            lines.append(f"{content}")
        return "\n".join(lines)
