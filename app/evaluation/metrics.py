"""
app/evaluation/metrics.py
─────────────────────────────────────────────────────────────────────────────
Evaluation framework wrapping RAGAS metrics.

Metrics computed:
  faithfulness       — Is the answer grounded in the retrieved context?
  answer_relevance   — Does the answer address the question?
  context_precision  — Are retrieved chunks relevant to the question?
  context_recall     — Do retrieved chunks cover the ground-truth answer?

Usage:
    evaluator = RAGEvaluator()
    results   = await evaluator.evaluate(questions_with_ground_truth)
    print(results.summary())
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EvalQuestion:
    question:     str
    ground_truth: str
    answer:       str = ""
    contexts:     List[str] = field(default_factory=list)


@dataclass
class EvalResults:
    faithfulness:      Optional[float] = None
    answer_relevance:  Optional[float] = None
    context_precision: Optional[float] = None
    context_recall:    Optional[float] = None
    num_questions:     int = 0
    per_question:      List[dict] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"Evaluation Results ({self.num_questions} questions)"]
        for name, val in [
            ("Faithfulness",      self.faithfulness),
            ("Answer Relevance",  self.answer_relevance),
            ("Context Precision", self.context_precision),
            ("Context Recall",    self.context_recall),
        ]:
            score = f"{val:.3f}" if val is not None else "N/A"
            lines.append(f"  {name:<20} {score}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "faithfulness":      self.faithfulness,
            "answer_relevance":  self.answer_relevance,
            "context_precision": self.context_precision,
            "context_recall":    self.context_recall,
            "num_questions":     self.num_questions,
        }


class RAGEvaluator:
    """Runs end-to-end RAG evaluation using RAGAS."""

    def __init__(self, pipeline=None):
        self._pipeline = pipeline

    async def evaluate(self, questions: List[EvalQuestion]) -> EvalResults:
        """
        Run the RAG pipeline on each question and compute RAGAS metrics.
        """
        pipeline = self._get_pipeline()

        # Generate answers for all questions
        for q in questions:
            try:
                answer = await pipeline.run(q.question)
                q.answer   = answer.answer_text
                q.contexts = [c.excerpt for c in answer.citations]
            except Exception as exc:
                logger.warning("Pipeline failed for question '%s': %s", q.question[:60], exc)
                q.answer   = ""
                q.contexts = []

        return self._compute_metrics(questions)

    def evaluate_from_file(self, path: str | Path) -> EvalResults:
        """
        Load ground-truth questions from a JSONL file and evaluate synchronously.
        Each line: {"question": "...", "ground_truth": "..."}
        """
        items = []
        for line in Path(path).read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                items.append(EvalQuestion(
                    question=d["question"],
                    ground_truth=d.get("ground_truth", ""),
                ))
        return asyncio.run(self.evaluate(items))

    def _compute_metrics(self, questions: List[EvalQuestion]) -> EvalResults:
        results = EvalResults(num_questions=len(questions))
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness, answer_relevancy,
                context_precision, context_recall,
            )
            from datasets import Dataset

            data = Dataset.from_list([
                {
                    "question":    q.question,
                    "ground_truth": q.ground_truth,
                    "answer":      q.answer,
                    "contexts":    q.contexts,
                }
                for q in questions
            ])
            scores = evaluate(
                data,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            )
            results.faithfulness      = float(scores.get("faithfulness",      0))
            results.answer_relevance  = float(scores.get("answer_relevancy",  0))
            results.context_precision = float(scores.get("context_precision", 0))
            results.context_recall    = float(scores.get("context_recall",    0))
            results.per_question      = scores.to_pandas().to_dict(orient="records")

        except ImportError:
            logger.warning("RAGAS not installed. Install: pip install ragas")
        except Exception as exc:
            logger.error("RAGAS evaluation failed: %s", exc)

        return results

    def _get_pipeline(self):
        if self._pipeline is None:
            from app.rag.pipeline import RAGPipeline
            self._pipeline = RAGPipeline()
        return self._pipeline
