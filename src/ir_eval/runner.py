"""Evaluation runner — orchestrates golden set evaluation.

Loads a golden set, iterates queries through an adapter, computes
per-query metrics, and aggregates into an EvalRun.

Usage:
    from ir_eval.runner import run_evaluation

    run = run_evaluation(golden_set, adapter, top_k=10)
    print(run.metrics)
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime

from ir_eval.adapter import RetrievalAdapter
from ir_eval.metrics.ranking import (
    aggregate_hit_rate,
    average_precision,
    hit_at_k,
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
)
from ir_eval.types import (
    EvalRun,
    GoldenSet,
    QueryResult,
    ResultSet,
    RetrievedItem,
)


def _evaluate_query(
    query_text: str,
    relevant_ids: set[str],
    retrieved: list[RetrievedItem],
    top_k: int,
    relevance_grades: dict[str, int] | None = None,
) -> dict[str, float | bool]:
    """Compute per-query metrics from retrieved items.

    Args:
        query_text: The query string (for error context).
        relevant_ids: Set of known relevant document IDs.
        retrieved: Ranked list of retrieved items.
        top_k: Cutoff for hit/precision metrics.
        relevance_grades: Optional graded relevance mapping.

    Returns:
        Dict with hit, reciprocal_rank, ndcg, precision_at_k, average_precision.
    """
    retrieved_ids = [item.id for item in retrieved]

    return {
        "hit": hit_at_k(retrieved_ids, relevant_ids, k=top_k),
        "reciprocal_rank": reciprocal_rank(retrieved_ids, relevant_ids),
        "ndcg": ndcg_at_k(retrieved_ids, relevant_ids, k=top_k, relevance_grades=relevance_grades),
        "precision_at_k": precision_at_k(retrieved_ids, relevant_ids, k=top_k),
        "average_precision": average_precision(retrieved_ids, relevant_ids),
    }


def run_evaluation(
    golden_set: GoldenSet,
    adapter: RetrievalAdapter,
    top_k: int = 10,
) -> EvalRun:
    """Run a full evaluation of a retrieval adapter against a golden set.

    Args:
        golden_set: The golden set with queries and known relevant docs.
        adapter: The retrieval system to evaluate.
        top_k: Number of results to retrieve per query.

    Returns:
        EvalRun with per-query results and aggregate metrics.

    Raises:
        RuntimeError: If the adapter fails on any query.
    """
    query_results: list[QueryResult] = []

    for golden_query in golden_set.queries:
        relevant_ids = set(golden_query.relevant_ids)
        grades = golden_query.relevance_grades if golden_query.relevance_grades else None

        # Time the retrieval
        t0 = time.perf_counter()
        retrieved = adapter.retrieve(golden_query.query, top_k=top_k)
        latency_ms = (time.perf_counter() - t0) * 1000

        # Compute per-query metrics
        metrics = _evaluate_query(
            query_text=golden_query.query,
            relevant_ids=relevant_ids,
            retrieved=retrieved,
            top_k=top_k,
            relevance_grades=grades,
        )

        qr = QueryResult(
            query=golden_query,
            retrieved=tuple(retrieved),
            hit=bool(metrics["hit"]),
            reciprocal_rank=float(metrics["reciprocal_rank"]),
            ndcg=float(metrics["ndcg"]),
            precision_at_k=float(metrics["precision_at_k"]),
            average_precision=float(metrics["average_precision"]),
            latency_ms=latency_ms,
        )
        query_results.append(qr)

    # Aggregate metrics
    n = len(query_results) if query_results else 0
    ndcg_avg = sum(qr.ndcg for qr in query_results) / n if n else 0.0
    prec_avg = sum(qr.precision_at_k for qr in query_results) / n if n else 0.0
    lat_avg = sum(qr.latency_ms or 0.0 for qr in query_results) / n if n else 0.0

    aggregate = {
        "hit_rate": aggregate_hit_rate([qr.hit for qr in query_results]),
        "mrr": mean_reciprocal_rank([qr.reciprocal_rank for qr in query_results]),
        "ndcg_at_k": ndcg_avg,
        "precision_at_k": prec_avg,
        "map": mean_average_precision([qr.average_precision for qr in query_results]),
        "avg_latency_ms": lat_avg,
    }

    run_id = f"run-{uuid.uuid4().hex[:12]}"

    return EvalRun(
        id=run_id,
        golden_set_name=golden_set.name,
        adapter_name=adapter.name,
        timestamp=datetime.now(UTC),
        query_results=tuple(query_results),
        metrics=aggregate,
        config={"top_k": top_k, "golden_set_version": golden_set.version},
    )


def evaluate_from_results(
    golden_set: GoldenSet,
    result_set: ResultSet,
    *,
    top_k: int = 10,
) -> EvalRun:
    """Evaluate pre-computed retrieval results against a golden set.

    This is the primary ``ir-eval`` evaluation path: pipe your retrieval
    system's output into a JSON file, load it as a ``ResultSet``, and
    evaluate without needing a live adapter.

    Args:
        golden_set: The golden set with queries and known relevant docs.
        result_set: Pre-computed retrieval results.
        top_k: Cutoff for hit/precision metrics.

    Returns:
        EvalRun with per-query results and aggregate metrics.

    Raises:
        ValueError: If a golden query has no matching result in the result set.
    """
    query_results: list[QueryResult] = []

    for golden_query in golden_set.queries:
        relevant_ids = set(golden_query.relevant_ids)
        grades = golden_query.relevance_grades if golden_query.relevance_grades else None

        retrieved = result_set.lookup(golden_query.query)
        if not retrieved:
            raise ValueError(
                f"No results found for query '{golden_query.query}' in result set '{result_set.name}'"
            )

        metrics = _evaluate_query(
            query_text=golden_query.query,
            relevant_ids=relevant_ids,
            retrieved=retrieved,
            top_k=top_k,
            relevance_grades=grades,
        )

        qr = QueryResult(
            query=golden_query,
            retrieved=tuple(retrieved),
            hit=bool(metrics["hit"]),
            reciprocal_rank=float(metrics["reciprocal_rank"]),
            ndcg=float(metrics["ndcg"]),
            precision_at_k=float(metrics["precision_at_k"]),
            average_precision=float(metrics["average_precision"]),
        )
        query_results.append(qr)

    n = len(query_results) if query_results else 0
    ndcg_avg = sum(qr.ndcg for qr in query_results) / n if n else 0.0
    prec_avg = sum(qr.precision_at_k for qr in query_results) / n if n else 0.0

    aggregate = {
        "hit_rate": aggregate_hit_rate([qr.hit for qr in query_results]),
        "mrr": mean_reciprocal_rank([qr.reciprocal_rank for qr in query_results]),
        "ndcg_at_k": ndcg_avg,
        "precision_at_k": prec_avg,
        "map": mean_average_precision([qr.average_precision for qr in query_results]),
    }

    run_id = f"run-{uuid.uuid4().hex[:12]}"

    return EvalRun(
        id=run_id,
        golden_set_name=golden_set.name,
        adapter_name=result_set.name,
        timestamp=datetime.now(UTC),
        query_results=tuple(query_results),
        metrics=aggregate,
        config={"top_k": top_k, "golden_set_version": golden_set.version, "source": "result_set"},
    )
