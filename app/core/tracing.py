"""
app/core/tracing.py
─────────────────────────────────────────────────────────────────────────────
Langfuse observability wrapper.

Design principles:
  - If Langfuse is not configured → silent no-op, zero crashes, zero overhead
  - All trace/span context flows via Python contextvars (async-safe)
  - One public API: `get_tracer()` returns the singleton (real or no-op)
  - Every RAG pipeline step and agent tool call creates a span

Langfuse concepts used here:
  Trace      → one user-facing request (ask, ingest, agent run)
  Span       → one pipeline step within a trace
  Generation → an LLM call (tracks tokens, model, prompt, completion)

Usage:
    tracer = get_tracer()

    async with tracer.trace("rag_pipeline", input={"question": q}) as t:
        async with t.span("retrieval", input={"query": q, "top_k": k}) as s:
            chunks = await retriever.retrieve(q, k)
            s.end(output={"chunk_count": len(chunks)})

        async with t.generation("llm_answer", model=model, prompt=prompt) as g:
            answer = await llm.generate(prompt)
            g.end(completion=answer)
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any, AsyncGenerator, Dict, Optional

logger = logging.getLogger(__name__)

# Context variable carrying the current active trace id (for log correlation)
_current_trace_id: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)


def get_trace_id() -> Optional[str]:
    """Return the active trace id for the current async context (for log injection)."""
    return _current_trace_id.get()


# ─────────────────────────── No-op implementations ───────────────────────────

class _NoOpSpan:
    """Silent no-op span — all methods are valid but do nothing."""
    def __init__(self, name: str):
        self.name = name
        self._start = time.perf_counter()

    def end(self, output: Any = None, status: str = "success", **kwargs) -> None:
        pass

    def update(self, **kwargs) -> None:
        pass

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000


class _NoOpGeneration(_NoOpSpan):
    def end(self, completion: str = "", output: Any = None, **kwargs) -> None:
        pass


class _NoOpTrace:
    def __init__(self, name: str, trace_id: str):
        self.id   = trace_id
        self.name = name

    @asynccontextmanager
    async def span(self, name: str, input: Any = None, **kwargs) -> AsyncGenerator[_NoOpSpan, None]:
        s = _NoOpSpan(name)
        try:
            yield s
        finally:
            s.end()

    @asynccontextmanager
    async def generation(
        self, name: str, model: str = "", prompt: str = "", **kwargs
    ) -> AsyncGenerator[_NoOpGeneration, None]:
        g = _NoOpGeneration(name)
        try:
            yield g
        finally:
            g.end()

    def end(self, output: Any = None, status: str = "success") -> None:
        pass


class _NoOpTracer:
    """Returned when Langfuse is not configured."""

    @asynccontextmanager
    async def trace(
        self, name: str, input: Any = None, user_id: str = None, **kwargs
    ) -> AsyncGenerator[_NoOpTrace, None]:
        import uuid
        tid   = str(uuid.uuid4())
        token = _current_trace_id.set(tid)
        t = _NoOpTrace(name, tid)
        try:
            yield t
        finally:
            t.end()
            _current_trace_id.reset(token)


# ─────────────────────────── Real Langfuse tracer ────────────────────────────

class _LangfuseTracer:
    """Real Langfuse tracer — only instantiated when keys are configured."""

    def __init__(self, client):
        self._client = client

    @asynccontextmanager
    async def trace(
        self,
        name:    str,
        input:   Any = None,
        user_id: Optional[str] = None,
        **kwargs,
    ) -> AsyncGenerator[_LangfuseTrace, None]:
        lf_trace = self._client.trace(name=name, input=input, user_id=user_id, **kwargs)
        token    = _current_trace_id.set(lf_trace.id)
        wrapper  = _LangfuseTrace(lf_trace)
        try:
            yield wrapper
        except Exception as exc:
            lf_trace.update(status="error", status_message=str(exc))
            raise
        finally:
            wrapper.end()
            _current_trace_id.reset(token)


class _LangfuseTrace:
    def __init__(self, lf_trace):
        self._t  = lf_trace
        self.id  = lf_trace.id
        self.name = lf_trace.name

    @asynccontextmanager
    async def span(self, name: str, input: Any = None, **kwargs) -> AsyncGenerator[_LangfuseSpan, None]:
        lf_span = self._t.span(name=name, input=input, **kwargs)
        wrapper = _LangfuseSpan(lf_span)
        t_start = time.perf_counter()
        try:
            yield wrapper
        except Exception as exc:
            lf_span.end(status="error", status_message=str(exc))
            raise
        finally:
            if not wrapper._ended:
                lf_span.end()

    @asynccontextmanager
    async def generation(
        self,
        name:   str,
        model:  str = "",
        prompt: str = "",
        **kwargs,
    ) -> AsyncGenerator[_LangfuseGeneration, None]:
        lf_gen  = self._t.generation(name=name, model=model, input=prompt, **kwargs)
        wrapper = _LangfuseGeneration(lf_gen)
        try:
            yield wrapper
        except Exception as exc:
            lf_gen.end(status="error", status_message=str(exc))
            raise
        finally:
            if not wrapper._ended:
                lf_gen.end()

    def end(self, output: Any = None, status: str = "success") -> None:
        try:
            self._t.update(output=output, status=status)
        except Exception:
            pass


class _LangfuseSpan:
    def __init__(self, lf_span):
        self._s     = lf_span
        self._ended = False
        self._start = time.perf_counter()

    def end(self, output: Any = None, status: str = "success", **kwargs) -> None:
        if not self._ended:
            self._s.end(output=output, status=status, **kwargs)
            self._ended = True

    def update(self, **kwargs) -> None:
        self._s.update(**kwargs)

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000


class _LangfuseGeneration(_LangfuseSpan):
    def end(self, completion: str = "", output: Any = None, usage: dict = None, **kwargs) -> None:
        if not self._ended:
            self._s.end(
                output=completion or output,
                usage=usage,
                **kwargs,
            )
            self._ended = True


# ─────────────────────────── Singleton factory ───────────────────────────────

_tracer_instance: Optional[_NoOpTracer | _LangfuseTracer] = None


def get_tracer() -> _NoOpTracer | _LangfuseTracer:
    """
    Return the singleton tracer.
    On first call, connects to Langfuse if credentials are configured;
    otherwise returns the silent no-op tracer.
    Thread-safe: worst case two instances are created at startup; both are valid.
    """
    global _tracer_instance
    if _tracer_instance is not None:
        return _tracer_instance

    from app.config import get_settings
    s = get_settings()

    if s.langfuse_enabled:
        try:
            from langfuse import Langfuse
            client = Langfuse(
                public_key=s.langfuse_public_key,
                secret_key=s.langfuse_secret_key,
                host=s.langfuse_host,
            )
            _tracer_instance = _LangfuseTracer(client)
            logger.info("Langfuse tracing enabled → %s", s.langfuse_host)
        except Exception as exc:
            logger.warning("Langfuse init failed (%s) — using no-op tracer.", exc)
            _tracer_instance = _NoOpTracer()
    else:
        logger.debug("Langfuse not configured — using no-op tracer.")
        _tracer_instance = _NoOpTracer()

    return _tracer_instance


def reset_tracer() -> None:
    """Force re-initialisation (useful in tests)."""
    global _tracer_instance
    _tracer_instance = None
