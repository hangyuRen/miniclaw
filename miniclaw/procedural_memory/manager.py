"""Procedural memory manager: LLM-driven evaluate, retrieve (hybrid), and merge."""

from __future__ import annotations

import json

from loguru import logger

from miniclaw.config.schema import ProceduralMemoryConfig
from miniclaw.procedural_memory.store import Procedure, ProceduralMemoryStore

_RRF_K = 60  # standard RRF constant


# ---------------------------------------------------------------------------
# LLM prompt templates
# ---------------------------------------------------------------------------

_EVALUATE_PROMPT = """\
You are a procedural memory curator. A user just completed a task with an AI assistant.
Your job: decide if this task execution is worth saving as a reusable procedure.

## Criteria for saving
Save ONLY if ALL of these are true:
1. The task is REPEATABLE — another person could encounter the same kind of task.
2. The steps are GENERALIZABLE — not tied to one-time specifics (e.g. a specific file path or date).
3. The process took real effort (multiple tool calls or non-trivial reasoning).

Do NOT save:
- Simple Q&A or factual lookups.
- One-off personal tasks with no reusable pattern.
- Tasks that are trivially obvious.

## Task conversation
{conversation}

## Output format (strict JSON, no markdown fences)
If worth saving:
{{
  "save": true,
  "title": "<short imperative title, e.g. 'Parse PDF and extract tables'>",
  "description": "<1-2 sentences: what problem this solves and when to use it>",
  "steps": "<markdown numbered list of concrete, generalizable steps>",
  "tags": ["<tag1>", "<tag2>"]
}}

If not worth saving:
{{"save": false, "reason": "<one sentence>"}}
"""

_USER_PROFILE_PROMPT = """\
You are a user profile curator. A user just had a conversation with an AI assistant.
Your job: identify any facts worth adding to the user's long-term profile (MEMORY.md).

## What belongs in the user profile
- Identity: name, occupation, location, languages spoken
- Preferences: communication style, tool choices, output format preferences, topics of interest
- Relationships: people, teams, or projects the user often refers to
- Recurring constraints: timezone, hardware, access restrictions

## What does NOT belong
- Transient task details (file paths specific to this task, one-off requests)
- Procedural steps (those go into procedural memory instead)
- Facts already obviously in the existing profile (avoid duplicates)

## Existing MEMORY.md
{existing_memory}

## Conversation
{conversation}

## Output format (strict JSON, no markdown fences)
If there is new profile information worth adding:
{{
  "update": true,
  "facts": ["<concise fact 1>", "<concise fact 2>"]
}}

If nothing new to add:
{{"update": false}}
"""

_MEMORY_MERGE_PROMPT = """\
You are updating a user's long-term profile file (MEMORY.md).

## Current MEMORY.md content
{existing_memory}

## New facts to integrate
{new_facts}

## Instructions
- Integrate the new facts naturally into the existing sections (User Information, Preferences, Project Context, Important Notes).
- Add new section headings only if none of the existing sections fit.
- Do not duplicate existing information.
- Keep the file concise — remove or merge redundant lines.
- Preserve the existing markdown structure and the footer line.

Return the complete updated MEMORY.md content (no fences, raw markdown only).
"""

_MERGE_PROMPT = """\
You are a procedural memory curator merging two versions of a procedure.

## Existing procedure (older)
Title: {old_title}
Description: {old_description}
Steps:
{old_steps}

## New procedure (from recent task)
Title: {new_title}
Description: {new_description}
Steps:
{new_steps}

## Your job
Produce ONE merged procedure that is more complete and accurate than either.
- Prefer the new steps if they are more detailed or correct.
- Keep steps from the old version if they cover edge cases the new one misses.
- Merge descriptions if both add value.
- Merge tags (deduplicate).

## Output format (strict JSON, no markdown fences)
{{
  "title": "<merged title>",
  "description": "<merged description>",
  "steps": "<merged markdown numbered list>",
  "tags": ["<tag1>", "<tag2>"]
}}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_conversation_text(session_messages: list[dict]) -> str:
    lines = []
    for msg in session_messages:
        role = msg.get("role", "")
        content = _normalize_content(msg.get("content"))
        if role == "user":
            lines.append(f"USER: {content[:800]}")
        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
                lines.append(f"ASSISTANT (tool calls): {', '.join(names)}")
                if content:
                    lines.append(f"  reasoning: {content[:300]}")
            else:
                lines.append(f"ASSISTANT: {content[:800]}")
        elif role == "tool":
            name = msg.get("name", "tool")
            lines.append(f"TOOL [{name}]: {content[:400]}")
    return "\n".join(lines)

def _normalize_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    texts.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    texts.append("[image]")
            else:
                texts.append(str(item))
        return " ".join(texts)
    return str(content)

def _parse_json_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text)


def _rrf_fuse(
    vector_hits: list[tuple[int, float]],   # (proc_id, cosine_sim desc)
    bm25_hits: list[tuple[int, float]],     # (proc_id, bm25_score asc — lower=better)
    top_k: int,
    thresh: int
) -> list[int]:
    """Reciprocal Rank Fusion: returns proc_ids sorted by combined RRF score desc."""
    scores: dict[int, float] = {}
    for rank, (pid, _) in enumerate(vector_hits, start=1):
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (_RRF_K + rank)
    for rank, (pid, _) in enumerate(bm25_hits, start=1):
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (_RRF_K + rank)
    return [pid for pid, sc in sorted(scores.items(), key=lambda x: x[1], reverse=True) if sc > thresh][:top_k]


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class ProceduralMemoryManager:
    """
    Manages procedural memory with hybrid retrieval (vector + BM25 + RRF).

    Embedding is generated via litellm.aembedding() using the configured model.
    Vectors are stored as struct-packed float32 BLOBs in procedures_vec.
    All operations are scoped by user_id.
    """

    def __init__(
        self,
        config: ProceduralMemoryConfig,
        provider,           # LLMProvider — for LLM evaluation/merge calls
        default_model: str,
    ):
        self.config = config
        self.provider = provider
        self.llm_model = config.llm_model or default_model
        self.embed_model = config.embed_model  # e.g. "openai/text-embedding-3-small"
        self.api_key = config.api_key
        self.api_base = config.api_base
        self.store = ProceduralMemoryStore(config.db_path, embed_dim=config.embed_dim)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------
    async def _embed(self, text: str) -> list[float] | None:
        """Generate an embedding via litellm. Returns None on failure."""
        if not self.embed_model:
            return None
        try:
            import litellm
            resp = await litellm.aembedding(model=self.embed_model, input=[text], api_key = self.api_key, api_base = self.api_base)
            return resp.data[0]["embedding"]
        except Exception as e:
            logger.warning("Procedural memory: embedding failed ({}): {}", self.embed_model, e)
            return None

    async def _ensure_embeddings(self, user_id: str) -> None:
        """Compute and store embeddings for any procedure missing one."""
        if not self.embed_model:
            return
        pending = self.store.get_pending_embedding(user_id)
        if not pending:
            return
        for proc_id, text in pending:
            vec = await self._embed(text)
            if vec:
                self.store.store_embedding(proc_id, vec)
                logger.debug("Procedural memory: stored embedding for proc #{}", proc_id)

    # ------------------------------------------------------------------
    # Retrieval (hybrid BM25 + vector + RRF)
    # ------------------------------------------------------------------

    async def retrieve(self, user_message: str, user_id: str = "shared", thresh: int = 0.5) -> list[Procedure]:
        """
        Hybrid search: BM25 (FTS5) + vector (cosine) fused with RRF.
        Falls back to BM25-only when embedding model is not configured.
        """
        if not self.config.enabled:
            return []

        candidate_k = self.config.top_k * 3  # each arm fetches 3× for RRF headroom

        # --- BM25 arm (always available) ---
        bm25_query = " ".join(user_message.split()[:30])
        bm25_hits = self.store.bm25_search(bm25_query, user_id=user_id, top_k=candidate_k)

        # --- Vector arm (requires embed_model) ---
        vector_hits: list[tuple[int, float]] = []
        if self.embed_model:
            await self._ensure_embeddings(user_id)
            query_vec = await self._embed(user_message)
            if query_vec:
                vector_hits = self.store.vector_search(query_vec, user_id=user_id, top_k=candidate_k)

        if not bm25_hits and not vector_hits:
            return []

        # --- RRF fusion ---
        thresh = max(self.config.thresh, 0.5)
        fused_ids = _rrf_fuse(vector_hits, bm25_hits, top_k=self.config.top_k, thresh=thresh)

        # --- Fetch full Procedure objects ---
        proc_map = self.store.get_by_ids(fused_ids)
        results = [proc_map[pid] for pid in fused_ids if pid in proc_map]

        if results:
            logger.debug(
                "Procedural memory: hybrid retrieved {} procedure(s) for user='{}' (vec={}, bm25={})",
                len(results), user_id, len(vector_hits), len(bm25_hits),
            )
        return results

    def render_block(self, procedures: list[Procedure]) -> str | None:
        """Render retrieved procedures into a system-prompt block."""
        if not procedures:
            return None
        parts = ["# Procedural Memory\n\nRelevant procedures from past experience:\n"]
        for p in procedures:
            tags_str = ", ".join(p.tags) if p.tags else "—"
            parts.append(
                f"## {p.title}\n"
                f"**When to use:** {p.description}\n"
                f"**Tags:** {tags_str}\n\n"
                f"{p.steps}"
            )
        return "\n\n---\n\n".join(parts)

    # ------------------------------------------------------------------
    # Evaluation & storage (post-task, fire-and-forget)
    # ------------------------------------------------------------------

    async def evaluate_and_store(self, session_messages: list[dict], user_id: str = "shared") -> None:
        if not self.config.enabled:
            return
        try:
            await self._run_evaluate_and_store(session_messages, user_id)
        except Exception as e:
            logger.warning("Procedural memory evaluation failed: {}", e)

    async def _run_evaluate_and_store(self, session_messages: list[dict], user_id: str) -> None:
        conversation_text = _build_conversation_text(session_messages)
        if not isinstance(conversation_text, str):
            conversation_text = str(conversation_text)
        if not conversation_text.strip():
            return

        # --- Step 1: Ask LLM whether to save ---
        eval_prompt = _EVALUATE_PROMPT.format(conversation=conversation_text)
        response = await self.provider.chat(
            messages=[
                {"role": "system", "content": "You are a procedural memory curator. Respond only with valid JSON."},
                {"role": "user", "content": eval_prompt},
            ],
            tools=[],
            model=self.llm_model,
            max_tokens=1024,
        )
        raw = (response.content or "").strip()
        if not raw:
            return

        data = _parse_json_response(raw)
        if not data.get("save"):
            logger.debug("Procedural memory: LLM decided not to save. Reason: {}", data.get("reason", ""))
            return

        new_title: str = data.get("title", "").strip()
        new_desc: str = data.get("description", "").strip()
        new_steps: str = data.get("steps", "").strip()
        new_tags: list[str] = data.get("tags", [])

        if not new_title or not new_steps:
            logger.warning("Procedural memory: LLM returned save=true but missing title/steps.")
            return

        # --- Step 2: Check for similar existing procedure (BM25 on title, same user) ---
        bm25_hits = self.store.bm25_search(new_title, user_id=user_id, top_k=1)
        existing = self.store.get_by_ids([pid for pid, _ in bm25_hits])
        old = next(iter(existing.values()), None) if existing else None

        if old is None:
            proc_id = self.store.insert(new_title, new_desc, new_steps, new_tags, user_id=user_id)
            # Embed the new procedure immediately so it's retrievable via vector search
            await self._ensure_embeddings(user_id)
            logger.info("Procedural memory: inserted #{} '{}' for user='{}'", proc_id, new_title, user_id)
            return

        # --- Step 3: Merge old and new via LLM ---
        merge_prompt = _MERGE_PROMPT.format(
            old_title=old.title,
            old_description=old.description,
            old_steps=old.steps,
            new_title=new_title,
            new_description=new_desc,
            new_steps=new_steps,
        )
        merge_response = await self.provider.chat(
            messages=[
                {"role": "system", "content": "You are a procedural memory curator. Respond only with valid JSON."},
                {"role": "user", "content": merge_prompt},
            ],
            tools=[],
            model=self.llm_model,
            max_tokens=1024,
        )
        merge_raw = (merge_response.content or "").strip()
        if not merge_raw:
            logger.warning("Procedural memory: merge LLM returned empty response; skipping update.")
            return

        merged = _parse_json_response(merge_raw)
        merged_title = merged.get("title", new_title).strip()
        merged_desc = merged.get("description", new_desc).strip()
        merged_steps = merged.get("steps", new_steps).strip()
        merged_tags = list({*old.tags, *merged.get("tags", new_tags)})

        self.store.update(old.id, merged_title, merged_desc, merged_steps, merged_tags)
        # Re-embed after merge (embedding_blob was reset to NULL by store.update)
        await self._ensure_embeddings(user_id)
        logger.info(
            "Procedural memory: merged #{} '{}' → '{}' for user='{}'",
            old.id, old.title, merged_title, user_id,
        )

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def bump_usage(self, proc_id: int) -> None:
        try:
            self.store.bump_usage(proc_id)
        except Exception as e:
            logger.warning("Procedural memory bump_usage failed: {}", e)

    # ------------------------------------------------------------------
    # User profile extraction → MEMORY.md
    # ------------------------------------------------------------------

    async def evaluate_user_profile(self, session_messages: list[dict], memory_store) -> None:
        """Background task: extract user profile facts and merge into MEMORY.md."""
        if not self.config.enabled:
            return
        try:
            await self._run_evaluate_user_profile(session_messages, memory_store)
        except Exception as e:
            logger.warning("User profile evaluation failed: {}", e)

    async def _run_evaluate_user_profile(self, session_messages: list[dict], memory_store) -> None:
        conversation_text = _build_conversation_text(session_messages)
        if not conversation_text.strip():
            return

        existing_memory = memory_store.read_long_term() or "(empty)"

        eval_prompt = _USER_PROFILE_PROMPT.format(
            existing_memory=existing_memory,
            conversation=conversation_text,
        )
        response = await self.provider.chat(
            messages=[
                {"role": "system", "content": "You are a user profile curator. Respond only with valid JSON."},
                {"role": "user", "content": eval_prompt},
            ],
            tools=[],
            model=self.llm_model,
            max_tokens=512,
        )
        raw = (response.content or "").strip()
        if not raw:
            return

        data = _parse_json_response(raw)
        if not data.get("update"):
            logger.debug("User profile: no new facts to add.")
            return

        new_facts: list[str] = data.get("facts", [])
        if not new_facts:
            return

        facts_text = "\n".join(f"- {f}" for f in new_facts)
        merge_prompt = _MEMORY_MERGE_PROMPT.format(
            existing_memory=existing_memory,
            new_facts=facts_text,
        )
        merge_response = await self.provider.chat(
            messages=[
                {"role": "system", "content": "You are updating a MEMORY.md file. Return only the updated file content."},
                {"role": "user", "content": merge_prompt},
            ],
            tools=[],
            model=self.llm_model,
            max_tokens=2048,
        )
        updated = (merge_response.content or "").strip()
        if not updated:
            logger.warning("User profile: merge LLM returned empty response; skipping.")
            return

        memory_store.write_long_term(updated)
        logger.info("User profile: MEMORY.md updated with {} new fact(s).", len(new_facts))
