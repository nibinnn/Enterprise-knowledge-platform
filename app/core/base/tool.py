"""
app/core/base/tool.py
─────────────────────────────────────────────────────────────────────────────
Abstract base class for all Agent tools.

Four tools are built on Days 16-17:
  - SearchTool         → wraps the RAG retriever
  - DocumentTool       → fetch / inspect a specific document
  - SummarizationTool  → multi-document summarization
  - AnalyticsTool      → cross-document aggregation & stats

Tools are invoked by the LLM through function/tool calling.
Each tool exposes a JSON schema (for the LLM) and a typed `run()` method.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ─────────────────────────── ToolResult ──────────────────────────────────────

class ToolResult(BaseModel):
    """
    Standardised return value for every tool.
    The agent loop consumes `output` as a string message.
    `data` holds structured data that the agent can reason over.
    """
    tool_name: str
    success: bool
    output: str                             # human/LLM-readable summary
    data: Optional[Dict[str, Any]] = None  # structured payload
    error: Optional[str] = None
    latency_ms: Optional[float] = None

    @classmethod
    def ok(cls, tool_name: str, output: str, data: Optional[Dict] = None, latency_ms: float = 0) -> "ToolResult":
        return cls(tool_name=tool_name, success=True, output=output, data=data, latency_ms=latency_ms)

    @classmethod
    def fail(cls, tool_name: str, error: str, latency_ms: float = 0) -> "ToolResult":
        return cls(tool_name=tool_name, success=False, output=f"Tool error: {error}", error=error, latency_ms=latency_ms)


# ─────────────────────────── BaseTool ────────────────────────────────────────

class BaseTool(ABC):
    """
    Every agent tool must implement:
      - name            → unique identifier used in function-calling schema
      - description     → shown to the LLM to decide when to call this tool
      - parameters_schema → JSON Schema for the tool's input parameters
      - _run()          → actual execution logic
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique snake_case tool name, e.g. 'search_knowledge_base'."""

    @property
    @abstractmethod
    def description(self) -> str:
        """One-paragraph description shown to the LLM."""

    @property
    @abstractmethod
    def parameters_schema(self) -> Dict[str, Any]:
        """
        JSON Schema object describing the tool's input parameters.
        Follows OpenAI / Anthropic tool-calling format:
        {
            "type": "object",
            "properties": { "query": {"type": "string", "description": "..."} },
            "required": ["query"]
        }
        """

    def run(self, **kwargs) -> ToolResult:
        """
        Public entry point. Wraps `_run()` with timing and error handling.
        Agents call this method — never `_run()` directly.
        """
        start = time.perf_counter()
        logger.info("[Tool:%s] called with args: %s", self.name, kwargs)
        try:
            result = self._run(**kwargs)
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.error("[Tool:%s] failed: %s", self.name, exc)
            return ToolResult.fail(self.name, str(exc), latency_ms=latency_ms)
        latency_ms = (time.perf_counter() - start) * 1000
        result.latency_ms = latency_ms
        logger.info("[Tool:%s] completed in %.1f ms, success=%s", self.name, latency_ms, result.success)
        return result

    async def arun(self, **kwargs) -> ToolResult:
        """
        Async version of `run()`. Override `_arun()` in async tools.
        Falls back to synchronous `_run()` if `_arun` is not overridden.
        """
        start = time.perf_counter()
        logger.info("[Tool:%s] async call with args: %s", self.name, kwargs)
        try:
            result = await self._arun(**kwargs)
        except NotImplementedError:
            result = self._run(**kwargs)
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            return ToolResult.fail(self.name, str(exc), latency_ms=latency_ms)
        result.latency_ms = (time.perf_counter() - start) * 1000
        return result

    @abstractmethod
    def _run(self, **kwargs) -> ToolResult:
        """Synchronous execution. Implement the tool logic here."""

    async def _arun(self, **kwargs) -> ToolResult:
        """Async execution. Override in I/O-bound tools."""
        raise NotImplementedError

    def to_llm_schema(self) -> Dict[str, Any]:
        """
        Return the tool definition in Anthropic / OpenAI function-calling format.
        Used by the agent orchestrator to register tools with the LLM.
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters_schema,    # Anthropic format
        }


# ─────────────────────────── ToolRegistry ────────────────────────────────────

class ToolRegistry:
    """
    Manages the set of tools available to the agent.
    The agent loop uses this to look up tools by name after the LLM
    decides which one to call.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            logger.warning("Tool '%s' is being overwritten in the registry.", tool.name)
        self._tools[tool.name] = tool
        logger.debug("Registered tool: '%s'", tool.name)

    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered. Available: {list(self._tools)}")
        return self._tools[name]

    def all_tools(self) -> List[BaseTool]:
        return list(self._tools.values())

    def llm_schemas(self) -> List[Dict[str, Any]]:
        """Return all tool schemas for injection into the LLM system prompt."""
        return [t.to_llm_schema() for t in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)
