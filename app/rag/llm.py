"""
app/rag/llm.py
─────────────────────────────────────────────────────────────────────────────
LLM client used by the RAG pipeline and agent layer.
Supports Claude (Anthropic) and GPT-4 (OpenAI) with streaming.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import logging
from typing import AsyncGenerator, List, Optional

from app.config import get_settings

logger   = logging.getLogger(__name__)
settings = get_settings()

_RAG_SYSTEM_PROMPT = """You are an expert knowledge assistant. Answer questions using ONLY the provided sources.

Rules:
1. Ground every claim in a specific source. Cite with [N] inline.
2. If the sources don't contain the answer, say "I don't have enough information in the provided sources."
3. Never invent facts not present in the sources.
4. Be concise but complete. Use bullet points for lists.
5. If multiple sources support a claim, cite all of them: [1][3].
"""

_AGENT_SYSTEM_PROMPT = """You are an intelligent research agent with access to a knowledge base.
You can use tools to search, retrieve, and analyse enterprise documents.
Think step by step. Use the minimum number of tool calls needed.
Always ground your final answer in retrieved evidence.
"""


class LLMClient:
    """
    Unified LLM interface for both Claude and OpenAI.
    Provider is selected from config (LLM_PROVIDER env var).
    """

    def __init__(
        self,
        provider:    Optional[str] = None,
        model:       Optional[str] = None,
        max_tokens:  Optional[int] = None,
        temperature: Optional[float] = None,
    ):
        self._provider    = provider    or settings.llm_provider.value
        self._model       = model       or settings.llm_model
        self._max_tokens  = max_tokens  or settings.llm_max_tokens
        self._temperature = temperature if temperature is not None else settings.llm_temperature
        self._client      = None

    # ── RAG answer generation ─────────────────────────────────────────────────

    async def generate_answer(
        self,
        question:       str,
        context:        str,
        system_prompt:  Optional[str] = None,
    ) -> str:
        """Generate a grounded answer from context."""
        prompt = (
            f"<sources>\n{context}\n</sources>\n\n"
            f"Question: {question}\n\n"
            "Answer (cite sources with [N]):"
        )
        return await self.complete(
            prompt, system=system_prompt or _RAG_SYSTEM_PROMPT
        )

    async def complete(self, prompt: str, system: Optional[str] = None) -> str:
        """Single completion call — returns the full response text."""
        if self._provider == "anthropic":
            return await self._claude_complete(prompt, system)
        return await self._openai_complete(prompt, system)

    async def stream(
        self, prompt: str, system: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Streaming completion — yields text deltas."""
        if self._provider == "anthropic":
            async for delta in self._claude_stream(prompt, system):
                yield delta
        else:
            async for delta in self._openai_stream(prompt, system):
                yield delta

    def get_tool_schema_format(self) -> str:
        """Return 'anthropic' or 'openai' so the agent formats tools correctly."""
        return self._provider

    # ── Claude ────────────────────────────────────────────────────────────────

    async def _claude_complete(self, prompt: str, system: Optional[str]) -> str:
        import anthropic
        client = self._get_client()
        messages = [{"role": "user", "content": prompt}]
        kwargs   = dict(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            messages=messages,
        )
        if system:
            kwargs["system"] = system
        response = await client.messages.create(**kwargs)
        return response.content[0].text

    async def _claude_stream(self, prompt: str, system: Optional[str]) -> AsyncGenerator[str, None]:
        import anthropic
        client = self._get_client()
        messages = [{"role": "user", "content": prompt}]
        kwargs   = dict(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            messages=messages,
        )
        if system:
            kwargs["system"] = system
        async with client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text

    # ── OpenAI ────────────────────────────────────────────────────────────────

    async def _openai_complete(self, prompt: str, system: Optional[str]) -> str:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = await client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        return response.choices[0].message.content or ""

    async def _openai_stream(self, prompt: str, system: Optional[str]) -> AsyncGenerator[str, None]:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        async with client.chat.completions.create(
            model=self._model, messages=messages,
            max_tokens=self._max_tokens, temperature=self._temperature, stream=True,
        ) as stream:
            async for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    yield delta

    # ── Client factory ────────────────────────────────────────────────────────

    def _get_client(self):
        if self._client is None:
            if self._provider == "anthropic":
                import anthropic
                self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
            else:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        return self._client
