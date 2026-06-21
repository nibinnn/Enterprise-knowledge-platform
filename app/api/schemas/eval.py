"""app/api/schemas/eval.py — evaluation framework schemas."""
from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EvalRunRequest(BaseModel):
    run_name: str = Field(..., min_length=1, max_length=200)
    dataset_name: str = Field(..., min_length=1, max_length=200)
    questions: Optional[List[Dict[str, str]]] = None  # [{"question": ..., "ground_truth": ...}]
    dataset_path: Optional[str] = None                # path to a JSONL ground-truth file


class EvalMetricsOut(BaseModel):
    faithfulness: Optional[float] = None
    answer_relevance: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None


class EvalRunOut(BaseModel):
    id: str
    run_name: str
    dataset_name: str
    metrics: EvalMetricsOut
    llm_model: Optional[str] = None
    embedding_model: Optional[str] = None
    chunking_strategy: Optional[str] = None
    num_questions: int = 0
    created_at: datetime

    model_config = {"from_attributes": True}
