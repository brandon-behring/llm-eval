"""Side-by-side comparison of two evaluation runs.

Produces per-metric deltas and per-query win/loss/tie analysis.
For statistical significance, see metrics.confidence and drift.detector.

Usage:
    from ir_eval.compare import compare_runs

    comparison = compare_runs(run_a, run_b)
    for metric, delta in comparison["metric_deltas"].items():
        print(f"{metric}: {delta:+.4f}")
"""

from __future__ import annotations

from typing import Any

from ir_eval.types import EvalRun


def compare_runs(run_a: EvalRun, run_b: EvalRun) -> dict[str, Any]:
    """Compare two evaluation runs.

    Args:
        run_a: First (typically baseline) run.
        run_b: Second (typically current) run.

    Returns:
        Dict with:
            - metric_deltas: {metric_name: b_value - a_value}
            - metric_pct_deltas: {metric_name: percent change}
            - per_query_changes: list of {query, a_hit, b_hit, rr_delta}
            - summary: {wins, losses, ties} based on hit status
    """
    # Metric-level comparison
    all_metrics = set(run_a.metrics.keys()) | set(run_b.metrics.keys())
    metric_deltas: dict[str, float] = {}
    metric_pct_deltas: dict[str, float | None] = {}

    for metric in sorted(all_metrics):
        a_val = run_a.metrics.get(metric, 0.0)
        b_val = run_b.metrics.get(metric, 0.0)
        delta = b_val - a_val
        metric_deltas[metric] = delta
        metric_pct_deltas[metric] = (delta / a_val * 100) if a_val != 0 else None

    # Per-query comparison (align by query text)
    a_by_query = {qr.query.query: qr for qr in run_a.query_results}
    b_by_query = {qr.query.query: qr for qr in run_b.query_results}

    per_query_changes: list[dict[str, Any]] = []
    wins = losses = ties = 0

    for query_text in sorted(set(a_by_query.keys()) | set(b_by_query.keys())):
        qr_a = a_by_query.get(query_text)
        qr_b = b_by_query.get(query_text)

        a_hit = qr_a.hit if qr_a else False
        b_hit = qr_b.hit if qr_b else False
        a_rr = qr_a.reciprocal_rank if qr_a else 0.0
        b_rr = qr_b.reciprocal_rank if qr_b else 0.0

        if b_hit and not a_hit:
            wins += 1
        elif a_hit and not b_hit:
            losses += 1
        else:
            ties += 1

        per_query_changes.append({
            "query": query_text,
            "a_hit": a_hit,
            "b_hit": b_hit,
            "rr_delta": b_rr - a_rr,
        })

    return {
        "run_a_id": run_a.id,
        "run_b_id": run_b.id,
        "run_a_adapter": run_a.adapter_name,
        "run_b_adapter": run_b.adapter_name,
        "metric_deltas": metric_deltas,
        "metric_pct_deltas": metric_pct_deltas,
        "per_query_changes": per_query_changes,
        "summary": {"wins": wins, "losses": losses, "ties": ties},
    }
