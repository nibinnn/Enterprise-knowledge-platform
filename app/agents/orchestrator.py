"""
app/agents/orchestrator.py
─────────────────────────────────────────────────────────────────────────────
Tool-calling agent orchestrator using the Anthropic tool-use API.

Loop:
  1. Send question + tool schemas to LLM
  2. LLM returns a tool_use block or a text answer
  3. If tool_use: execute the tool, append result, loop
  4. If text: extract final answer + citations, return AgentResponse

Max iterations: 10 (prevents infinite loops on confused queries).
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.core.base.tool import ToolRegistry, ToolResult
from app.core.tracing import get_tracer
from app.config import get_settings

logger   = logging.getLogger(__name__)
settings = get_settings()

MAX_ITERATIONS = 10

_SYSTEM = """You are an intelligent enterprise knowledge assistant with access to tools.
Your goal: answer the user's question accurately using the knowledge base.

Strategy:
1. Start with a search to find relevant information.
2. Use additional tools if the first search isn't sufficient.
3. If you need details from a specific document, use get_document.
4. Synthesize findings into a clear, cited answer.
5. Stop once you have enough information — don't over-search.

Always cite your sources with [Source: filename] in the final answer."""


class AgentStep:
    def __init__(self, step: int, tool_name: str, tool_input: dict, tool_output: str, latency_ms: float):
        self.step       = step
        self.tool_name  = tool_name
        self.tool_input = tool_input
        self.tool_output = tool_output
        self.latency_ms = latency_ms


class AgentResponse:
    def __init__(self, answer: str, steps: List[AgentStep], total_ms: float):
        self.answer_id   = str(uuid.uuid4())
        self.answer      = answer
        self.steps       = steps
        self.total_steps = len(steps)
        self.total_ms    = total_ms
        self.created_at  = datetime.utcnow()


class AgentOrchestrator:
    """
    Claude tool-calling agent.

    Args:
        registry: ToolRegistry with all available tools registered.
        max_iter: Maximum tool-call iterations before forcing a final answer.
    """

    def __init__(self, registry: Optional[ToolRegistry] = None, max_iter: int = MAX_ITERATIONS):
        self._registry = registry or self._default_registry()
        self._max_iter = max_iter

    async def run(self, question: str, filters: Optional[Dict[str, Any]] = None) -> AgentResponse:
        from app.core.metrics import (
            AGENT_REQUESTS_TOTAL, AGENT_ITERATIONS, AGENT_TOOL_CALLS_TOTAL,
        )
        tracer   = get_tracer()
        t_start  = time.perf_counter()
        messages = [{"role": "user", "content": question}]
        steps:   List[AgentStep] = []
        tools    = self._registry.llm_schemas()
        final_text = "I was unable to fully answer the question within the iteration limit."

        async with tracer.trace("agent_run", input={"question": question}) as trace:
            try:
                for iteration in range(self._max_iter):
                    async with trace.generation(
                        f"llm_iteration_{iteration + 1}",
                        model=settings.llm_model,
                        prompt=f"[iteration {iteration + 1}] {question}",
                    ) as gen:
                        response    = await self._call_llm(messages, tools)
                        stop_reason = response.stop_reason
                        gen.end(completion=str(stop_reason))

                    # ── Text-only answer — done ───────────────────────────────
                    if stop_reason == "end_turn":
                        final_text = self._extract_text(response)
                        break

                    # ── Tool use ──────────────────────────────────────────────
                    if stop_reason == "tool_use":
                        tool_results_content = []
                        for block in response.content:
                            if block.type != "tool_use":
                                continue
                            tool_name  = block.name
                            tool_input = block.input or {}
                            tool_id    = block.id

                            t_tool = time.perf_counter()
                            tool_status = "success"
                            async with trace.span(
                                f"tool_{tool_name}",
                                input={"tool": tool_name, "input": tool_input},
                            ) as span:
                                try:
                                    tool   = self._registry.get(tool_name)
                                    result: ToolResult = await tool.arun(**tool_input)
                                    output = result.output
                                    span.end(output={"success": result.success, "output_len": len(output)})
                                except KeyError:
                                    output      = f"Tool '{tool_name}' not found."
                                    tool_status = "error"
                                    span.end(status="error")
                                except Exception as exc:
                                    output      = f"Tool error: {exc}"
                                    tool_status = "error"
                                    span.end(status="error")

                            tool_ms = (time.perf_counter() - t_tool) * 1000
                            AGENT_TOOL_CALLS_TOTAL.labels(
                                tool_name=tool_name, status=tool_status
                            ).inc()

                            steps.append(AgentStep(
                                step=iteration + 1,
                                tool_name=tool_name,
                                tool_input=tool_input,
                                tool_output=output[:1000],
                                latency_ms=round(tool_ms, 1),
                            ))
                            logger.info(
                                "[Agent] Step %d: %s → %d chars in %.0f ms trace_id=%s",
                                iteration + 1, tool_name, len(output), tool_ms, trace.id,
                            )

                            tool_results_content.append({
                                "type":        "tool_result",
                                "tool_use_id": tool_id,
                                "content":     output,
                            })

                        messages.append({"role": "assistant", "content": response.content})
                        messages.append({"role": "user",      "content": tool_results_content})
                        continue

                    final_text = self._extract_text(response)
                    break

                total_ms = (time.perf_counter() - t_start) * 1000
                AGENT_REQUESTS_TOTAL.labels(status="success").inc()
                AGENT_ITERATIONS.observe(len(steps))
                logger.info(
                    "[Agent] Completed in %.0f ms, %d steps trace_id=%s",
                    total_ms, len(steps), trace.id,
                )
                trace.end(output={"steps": len(steps), "answer_length": len(final_text)})
                return AgentResponse(answer=final_text, steps=steps, total_ms=round(total_ms, 1))

            except Exception as exc:
                AGENT_REQUESTS_TOTAL.labels(status="error").inc()
                logger.error("[Agent] Failed: %s trace_id=%s", exc, trace.id)
                raise

    # ── LLM call ──────────────────────────────────────────────────────────────

    async def _call_llm(self, messages: list, tools: list):
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        return await client.messages.create(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            system=_SYSTEM,
            tools=tools,
            messages=messages,
        )

    @staticmethod
    def _extract_text(response) -> str:
        parts = []
        for block in response.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "\n".join(parts).strip()

    # ── Default registry ──────────────────────────────────────────────────────

    @staticmethod
    def _default_registry() -> ToolRegistry:
        from app.agents.tools.search_tool        import SearchTool
        from app.agents.tools.document_tool      import DocumentTool
        from app.agents.tools.summarization_tool import SummarizationTool
        from app.agents.tools.analytics_tool     import AnalyticsTool
        registry = ToolRegistry()
        registry.register(SearchTool())
        registry.register(DocumentTool())
        registry.register(SummarizationTool())
        registry.register(AnalyticsTool())
        return registry
