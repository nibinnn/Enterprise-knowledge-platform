"""
app/rag/pipeline.py
─────────────────────────────────────────────────────────────────────────────
Full RAG pipeline: query → retriever → reranker → context_builder → LLM → answer.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

from app.core.models.document import Answer, Citation, QueryContext
from app.rag.context_builder import ContextBuilder
from app.rag.llm import LLMClient
from app.config import get_settings

logger   = logging.getLogger(__name__)
settings = get_settings()


class RAGPipeline:
    """
    Orchestrates the full RAG query pipeline.

    Usage:
        pipeline = RAGPipeline()
        answer   = await pipeline.run(question="What is the refund policy?")
    """

    def __init__(
        self,
        retriever=None,
        context_builder: Optional[ContextBuilder] = None,
        llm:             Optional[LLMClient]      = None,
        top_k:           int   = 20,
        rerank_top_n:    int   = 5,
    ):
        self._retriever       = retriever
        self._context_builder = context_builder or ContextBuilder(max_tokens=settings.context_max_tokens)
        self._llm             = llm or LLMClient()
        self._top_k           = top_k
        self._rerank_top_n    = rerank_top_n

    async def run(
        self,
        question: str,
        filters:  Optional[Dict[str, Any]] = None,
        top_k:    Optional[int] = None,
        rerank_top_n: Optional[int] = None,
    ) -> Answer:
        t_start = time.perf_counter()

        # 1. Retrieve + rerank
        t_ret = time.perf_counter()
        retriever = self._get_retriever()
        chunks    = await retriever.retrieve(
            query=question,
            top_k=top_k or self._top_k,
            filters=filters,
        )
        retrieval_ms = (time.perf_counter() - t_ret) * 1000

        # 2. Build context
        ctx = self._context_builder.build(question, chunks[:rerank_top_n or self._rerank_top_n])

        # 3. Generate answer
        t_llm = time.perf_counter()
        answer_text = await self._llm.generate_answer(
            question=question,
            context=ctx.formatted_context,
        )
        llm_ms = (time.perf_counter() - t_llm) * 1000

        # 4. Extract citations
        citations = self._extract_citations(answer_text, ctx)

        total_ms = (time.perf_counter() - t_start) * 1000
        logger.info("RAG complete in %.0f ms (retrieval=%.0f, llm=%.0f)", total_ms, retrieval_ms, llm_ms)

        return Answer(
            id=str(uuid.uuid4()),
            query=question,
            answer_text=answer_text,
            citations=citations,
            model_used=self._llm._model,
            retrieval_latency_ms=round(retrieval_ms, 1),
            llm_latency_ms=round(llm_ms, 1),
            total_latency_ms=round(total_ms, 1),
            created_at=datetime.utcnow(),
        )

    async def stream(
        self,
        question: str,
        filters:  Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[str, None]:
        """Streaming variant — yields text deltas then the citation block."""
        retriever = self._get_retriever()
        chunks    = await retriever.retrieve(question, self._top_k, filters)
        ctx       = self._context_builder.build(question, chunks[:self._rerank_top_n])

        full_text = ""
        async for delta in self._llm.stream(
            prompt=f"<sources>\n{ctx.formatted_context}\n</sources>\n\nQuestion: {question}\n\nAnswer:",
        ):
            full_text += delta
            yield delta

    def _extract_citations(self, answer_text: str, ctx: QueryContext) -> List[Citation]:
        """Extract [N] citation markers from answer text and map to source chunks."""
        import re
        cited_indices = set(int(m) for m in re.findall(r"\[(\d+)\]", answer_text))
        citations = []
        for idx in sorted(cited_indices):
            if 1 <= idx <= len(ctx.chunks):
                chunk = ctx.chunks[idx - 1]
                m = chunk.metadata
                citations.append(Citation(
                    chunk_id=chunk.chunk_id,
                    doc_id=m.doc_id,
                    doc_filename=m.doc_filename,
                    page_number=m.page_number,
                    section_heading=m.section_heading,
                    excerpt=chunk.text[:300] + ("…" if len(chunk.text) > 300 else ""),
                ))
        return citations

    def _get_retriever(self):
        if self._retriever is None:
            from app.retrieval.retriever import HybridRetriever
            self._retriever = HybridRetriever()
        return self._retriever


# ── Module-level singleton factory ────────────────────────────────────────────

def get_pipeline(**kwargs) -> RAGPipeline:
    return RAGPipeline(**kwargs)
