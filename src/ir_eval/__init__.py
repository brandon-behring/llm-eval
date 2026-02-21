"""ir-eval: Statistical RAG evaluation framework with drift detection.

Usage:
    from ir_eval import GoldenSet, GoldenQuery, EvalRun, RetrievalAdapter
    from ir_eval.runner import run_evaluation
    from ir_eval.metrics.ranking import mean_reciprocal_rank, ndcg_at_k
    from ir_eval.drift.detector import DriftDetector
"""

from ir_eval.adapter import RetrievalAdapter
from ir_eval.types import (
    Baseline,
    Difficulty,
    DriftResult,
    DriftSeverity,
    EvalRun,
    GoldenQuery,
    GoldenSet,
    QueryResult,
    ResultEntry,
    ResultSet,
    RetrievedItem,
)

__all__ = [
    "Baseline",
    "Difficulty",
    "DriftResult",
    "DriftSeverity",
    "EvalRun",
    "GoldenQuery",
    "GoldenSet",
    "QueryResult",
    "ResultEntry",
    "ResultSet",
    "RetrievalAdapter",
    "RetrievedItem",
]

__version__ = "0.1.0"
