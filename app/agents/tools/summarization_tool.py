"""app/agents/tools/summarization_tool.py — multi-document summarization."""
from __future__ import annotations
from app.core.base.tool import BaseTool, ToolResult


class SummarizationTool(BaseTool):
    """Summarize one or more documents or search results."""

    name        = "summarize"
    description = (
        "Generate a concise summary of retrieved content or a specific document. "
        "Use after searching to condense multiple results into a coherent overview. "
        "Supports 'brief' (2-3 sentences), 'standard' (1 paragraph), 'detailed' (bullet points)."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "content":  {"type": "string", "description": "The text content to summarize."},
            "style":    {"type": "string", "enum": ["brief", "standard", "detailed"], "default": "standard"},
            "focus":    {"type": "string", "description": "Optional topic to focus the summary on."},
        },
        "required": ["content"],
    }

    def __init__(self, llm=None):
        self._llm = llm

    def _run(self, content: str, style: str = "standard", focus: str = None) -> ToolResult:
        import asyncio
        try:
            return asyncio.run(self._arun(content=content, style=style, focus=focus))
        except RuntimeError:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self._arun(content=content, style=style, focus=focus))

    async def _arun(self, content: str, style: str = "standard", focus: str = None) -> ToolResult:
        llm = self._get_llm()
        style_instructions = {
            "brief":    "in 2-3 sentences",
            "standard": "in one clear paragraph",
            "detailed": "as a bullet-point list covering all key points",
        }
        instruction = style_instructions.get(style, "in one clear paragraph")
        focus_clause = f" Focus specifically on: {focus}." if focus else ""
        prompt = (
            f"Summarize the following content {instruction}.{focus_clause}\n\n"
            f"Content:\n{content[:8000]}\n\nSummary:"
        )
        summary = await llm.complete(prompt, system="You are a precise document summarizer. Be factual and concise.")
        return ToolResult.ok(self.name, summary.strip(), data={"style": style, "focus": focus})

    def _get_llm(self):
        if self._llm is None:
            from app.rag.llm import LLMClient
            self._llm = LLMClient()
        return self._llm
