"""Markdown report generator.

Produces markdown tables suitable for GitHub PRs, docs, and
commit messages.
"""

from __future__ import annotations

from typing import Any

from ir_eval.types import DriftResult, DriftSeverity, EvalRun


def report_eval_run(run: EvalRun) -> str:
    """Generate markdown report for an evaluation run.

    Args:
        run: The evaluation run.

    Returns:
        Markdown string.
    """
    lines = [
        f"# Evaluation Report: {run.id}",
        "",
        f"- **Adapter**: {run.adapter_name}",
        f"- **Golden Set**: {run.golden_set_name}",
        f"- **Timestamp**: {run.timestamp.isoformat()}",
        f"- **Queries**: {len(run.query_results)}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]

    for metric, value in sorted(run.metrics.items()):
        if "latency" in metric:
            lines.append(f"| {metric} | {value:.1f} ms |")
        elif value <= 1.0:
            lines.append(f"| {metric} | {value:.4f} |")
        else:
            lines.append(f"| {metric} | {value:.2f} |")

    # Missed queries
    misses = [qr for qr in run.query_results if not qr.hit]
    if misses:
        lines.extend([
            "",
            "## Missed Queries",
            "",
        ])
        for qr in misses:
            lines.append(f"- {qr.query.query}")

    return "\n".join(lines)


def report_comparison(comparison: dict[str, Any]) -> str:
    """Generate markdown comparison report.

    Args:
        comparison: Output from compare_runs().

    Returns:
        Markdown string.
    """
    lines = [
        f"# Comparison: {comparison['run_a_adapter']} vs {comparison['run_b_adapter']}",
        "",
        "| Metric | Delta | % Change |",
        "|--------|-------|----------|",
    ]

    for metric, delta in sorted(comparison["metric_deltas"].items()):
        pct = comparison["metric_pct_deltas"].get(metric)
        pct_str = f"{pct:+.1f}%" if pct is not None else "N/A"
        emoji = "+" if delta >= 0 else ""
        lines.append(f"| {metric} | {emoji}{delta:.4f} | {pct_str} |")

    summary = comparison["summary"]
    lines.extend([
        "",
        f"**Summary**: {summary['wins']} wins, {summary['losses']} losses, {summary['ties']} ties",
    ])

    return "\n".join(lines)


def report_drift(drift_results: list[DriftResult]) -> str:
    """Generate markdown drift report.

    Args:
        drift_results: Per-metric drift results.

    Returns:
        Markdown string.
    """
    severity_emoji = {
        DriftSeverity.INFO: "OK",
        DriftSeverity.WARNING: "WARN",
        DriftSeverity.CRITICAL: "CRIT",
    }

    lines = [
        "# Drift Detection Report",
        "",
        "| Metric | Baseline | Current | Delta | p-value | Severity |",
        "|--------|----------|---------|-------|---------|----------|",
    ]

    for dr in drift_results:
        p_str = f"{dr.p_value:.4f}" if dr.p_value is not None else "-"
        sev = severity_emoji[dr.severity]
        lines.append(
            f"| {dr.metric_name} | {dr.baseline_value:.4f} | "
            f"{dr.current_value:.4f} | {dr.delta:+.4f} | {p_str} | **{sev}** |"
        )

    # Summary
    max_severity = max(
        (dr.severity for dr in drift_results),
        default=DriftSeverity.INFO,
        key=lambda s: list(DriftSeverity).index(s),
    )
    lines.extend([
        "",
        f"**Overall**: {severity_emoji[max_severity]}",
    ])

    return "\n".join(lines)
