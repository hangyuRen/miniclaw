"""Context builder for assembling agent prompts."""

import base64
import mimetypes
import platform
import time
from pathlib import Path
from typing import Any

from loguru import logger

from miniclaw.agent.memory import MemoryStore
from miniclaw.agent.skills import SkillsLoader
from miniclaw.procedural_memory import ProceduralMemoryManager


class ContextBuilder:
    """
    Builds the context (system prompt + messages) for the agent.
    
    Assembles bootstrap files, memory, skills, and conversation history
    into a coherent prompt for the LLM.
    """
    
    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md", "IDENTITY.md"]
    
    def __init__(
        self,
        workspace: Path,
        procedural_memory: ProceduralMemoryManager | None = None,  # ProceduralMemoryManager | None
    ):
        self.workspace = workspace
        self.memory = MemoryStore(workspace)
        self.skills = SkillsLoader(workspace)
        self.procedural_memory = procedural_memory
        self._file_cache: dict[str, tuple[tuple[int, int, int, int] | None, str]] = {}
        self.last_build_stats: dict[str, Any] = {}
        # Skills loaded lazily this session via read_file (skill name → tool_call_ids used to load it)
        self._loaded_skills: dict[str, set[str]] = {}

    def register_skill_load(self, skill_name: str, tool_call_id: str) -> None:
        """Record that a skill was loaded via read_file during this session."""
        self._loaded_skills.setdefault(skill_name, set()).add(tool_call_id)

    def clear_loaded_skills(self) -> None:
        """Reset loaded skills (call at session start if desired)."""
        self._loaded_skills.clear()
    
    def build_system_prompt(
        self,
        session_summary: str | None = None,
        procedural_memory_block: str | None = None,
    ) -> str:
        """
        Build the system prompt from bootstrap files, memory, and skills.

        system prompt构成: system->skills->retrieve->summary->history(long-term + today)
        
        Args:
            skill_names: Optional list of skills to include.
        
        Returns:
            Complete system prompt.
        """
        parts = []
        
        # Core identity
        parts.append(self._get_identity())
        
        # Bootstrap files
        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            parts.append(bootstrap)
        
        # Skills - progressive loading
        # 1. Always-loaded skills: include full content
        always_skills = self.skills.get_always_skills()
        if always_skills:
            always_content = self.skills.load_skills_for_context(always_skills)
            if always_content:
                parts.append(f"# Active Skills\n\n{always_content}")
        
        # 2. Available skills: only show summary (agent uses read_file to load)
        skills_summary = self.skills.build_skills_summary()
        if skills_summary:
            parts.append(f"""# Skills

The following skills extend your capabilities. To use a skill, read its SKILL.md file using the read_file tool.
Skills with available=\"false\" need dependencies installed first - you can try installing them with apt/brew.

{skills_summary}""")
            
        if procedural_memory_block:
            parts.append(procedural_memory_block)

        if session_summary:
            parts.append(f"# Session Rolling Summary\n\n{session_summary}")
        
        # Memory context
        memory = self.memory.get_memory_context()
        if memory:
            parts.append(f"# Memory\n\n{memory}")
        
        return "\n\n---\n\n".join(parts)
    
    def _get_identity(self) -> str:
        """Get the core identity section."""
        from datetime import datetime
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        workspace_path = str(self.workspace.expanduser().resolve())
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"
        
        return f"""# miniclaw 🐈

You are miniclaw, a helpful AI assistant. You have access to tools that allow you to:
- Read, write, and edit files
- Execute shell commands
- Search the web and fetch web pages
- Send messages to users on chat channels
- Spawn subagents for complex background tasks

## Current Time
{now}

## Runtime
{runtime}

## Workspace
Your workspace is at: {workspace_path}
- Memory files: {workspace_path}/memory/MEMORY.md
- Daily notes: {workspace_path}/memory/YYYY-MM-DD.md
- Custom skills: {workspace_path}/skills/{{skill-name}}/SKILL.md

IMPORTANT: When responding to direct questions or conversations, reply directly with your text response.
Only use the 'message' tool when you need to send a message to a specific chat channel (like WhatsApp).
For normal conversation, just respond with text - do not call the message tool.

Always be helpful, accurate, and concise. When using tools, explain what you're doing.
When remembering something, write to {workspace_path}/memory/MEMORY.md"""
    
    def _load_bootstrap_files(self) -> str:
        """Load all bootstrap files from workspace."""
        parts = []
        
        for filename in self.BOOTSTRAP_FILES:
            file_path = self.workspace / filename
            if file_path.exists():
                content = self._read_cached_text(file_path)
                parts.append(f"## {filename}\n\n{content}")
        
        return "\n\n".join(parts) if parts else ""
    
    def _read_cached_text(self, file_path: Path) -> str:
        """Read file with lightweight stat+content cache for hot-path prompt assembly."""
        key = str(file_path.resolve())
        if not file_path.exists():
            self._file_cache[key] = (None, "")
            return ""

        stat = file_path.stat()
        version = (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, stat.st_ino)
        cached = self._file_cache.get(key)
        if cached and cached[0] == version:
            return cached[1]

        content = file_path.read_text(encoding="utf-8")
        self._file_cache[key] = (version, content)
        return content

    async def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        session_summary: str | None = None,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        user_id: str = "shared",
    ) -> list[dict[str, Any]]:
        """
        Build the complete message list for an LLM call.

        上下文构成: system prompt(包含 system、skills、procedural memory、summary、MEMORY.md) -> history -> user content

        Args:
            history: Previous conversation messages.
            current_message: The new user message.
            media: Optional list of local file paths for images/media.
            channel: Current channel (telegram, feishu, etc.).
            chat_id: Current chat/user ID.
            user_id: Sender identity used to scope procedural memory retrieval.

        Returns:
            List of messages including system prompt.
        """
        build_started = time.perf_counter()
        messages = []

        # Procedural memory retrieval — hybrid (BM25 + vector + RRF), scoped to user_id
        procedural_block = None
        procedural_ms = 0.0
        procedural_count = 0
        if self.procedural_memory:
            proc_started = time.perf_counter()
            procs = await self.procedural_memory.retrieve(current_message, user_id=user_id)
            procedural_ms = (time.perf_counter() - proc_started) * 1000.0
            procedural_count = len(procs)
            if procs:
                for p in procs:
                    self.procedural_memory.bump_usage(p.id)
                procedural_block = self.procedural_memory.render_block(procs)

        system_prompt_started = time.perf_counter()
        system_prompt = self.build_system_prompt(
            session_summary=session_summary,
            procedural_memory_block=procedural_block,
        )
        system_prompt_ms = (time.perf_counter() - system_prompt_started) * 1000.0
        if channel and chat_id:
            system_prompt += f"\n\n## Current Session\nChannel: {channel}\nChat ID: {chat_id}"
        messages.append({"role": "system", "content": system_prompt})

        messages.extend(self._filter_skill_messages(history))

        user_content = self._build_user_content(current_message, media)
        messages.append({"role": "user", "content": user_content})

        total_ms = (time.perf_counter() - build_started) * 1000.0
        self.last_build_stats = {
            "procedural_ms": round(procedural_ms, 3),
            "system_prompt_ms": round(system_prompt_ms, 3),
            "total_ms": round(total_ms, 3),
            "procedural_count": procedural_count,
            "history_messages": len(history),
            "channel": channel or "",
            "chat_id": chat_id or "",
        }
        logger.debug(
            "Context build timing | procedural={:.2f}ms({}) system_prompt={:.2f}ms total={:.2f}ms history={}",
            procedural_ms,
            procedural_count,
            system_prompt_ms,
            total_ms,
            len(history),
        )

        return messages

    _SKILL_HISTORY_PREVIEW = 200  # chars to keep from a skill result in history

    def _filter_skill_messages(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Truncate tool-result content for SKILL.md reads already seen this session,
        so old skill content does not bloat subsequent context windows.
        """
        if not self._loaded_skills:
            return history

        skill_tool_ids: set[str] = set()
        for ids in self._loaded_skills.values():
            skill_tool_ids.update(ids)

        result: list[dict[str, Any]] = []
        for msg in history:
            if (
                msg.get("role") == "tool"
                and msg.get("tool_call_id") in skill_tool_ids
            ):
                content = msg.get("content") or ""
                if len(content) > self._SKILL_HISTORY_PREVIEW:
                    content = content[: self._SKILL_HISTORY_PREVIEW] + "...(truncated, skill already loaded)"
                    msg = {**msg, "content": content}
            result.append(msg)
        return result

    def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        """Build user message content with optional base64-encoded images."""
        if not media:
            return text
        
        images = []
        for path in media:
            p = Path(path)
            mime, _ = mimetypes.guess_type(path)
            if not p.is_file() or not mime or not mime.startswith("image/"):
                continue
            b64 = base64.b64encode(p.read_bytes()).decode()
            images.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        
        if not images:
            return text
        return images + [{"type": "text", "text": text}]
    
    def add_tool_result(
        self,
        messages: list[dict[str, Any]],
        tool_call_id: str,
        tool_name: str,
        result: str
    ) -> list[dict[str, Any]]:
        """
        Add a tool result to the message list.
        
        Args:
            messages: Current message list.
            tool_call_id: ID of the tool call.
            tool_name: Name of the tool.
            result: Tool execution result.
        
        Returns:
            Updated message list.
        """
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result
        })
        return messages
    
    def add_assistant_message(
        self,
        messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None
    ) -> list[dict[str, Any]]:
        """
        Add an assistant message to the message list.
        
        Args:
            messages: Current message list.
            content: Message content.
            tool_calls: Optional tool calls.
        
        Returns:
            Updated message list.
        """
        msg: dict[str, Any] = {"role": "assistant", "content": content or ""}
        
        if tool_calls:
            msg["tool_calls"] = tool_calls
        
        messages.append(msg)
        return messages
