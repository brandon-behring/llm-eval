"""Rich console reporter for evaluation results.

Displays colorful terminal tables with metrics, per-query results,
and drift alerts.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.table import Table

from ir_eval.types import DriftResult, DriftSeverity, EvalRun


def report_eval_run(run: EvalRun, console: Console | None = None) -> None:
    """Print evaluation run results as a rich table.

    Args:
        run: The evaluation run to display.
        console: Rich console (creates new one if None).
    """
    if console is None:
        console = Console()

    console.print(f"\n[bold]Evaluation Run:[/bold] {run.id}")
    console.print(f"  Adapter: {run.adapter_name}")
    console.print(f"  Golden Set: {run.golden_set_name}")
    console.print(f"  Timestamp: {run.timestamp.isoformat()}")
    console.print(f"  Queries: {len(run.query_results)}")

    # Metrics table
    table = Table(title="Aggregate Metrics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")

    for metric, value in sorted(run.metrics.items()):
        if "latency" in metric:
            table.add_row(metric, f"{value:.1f} ms")
        elif value <= 1.0:
            table.add_row(metric, f"{value:.4f}")
        else:
            table.add_row(metric, f"{value:.2f}")

    console.print(table)

    # Per-query summary
    hits = sum(1 for qr in run.query_results if qr.hit)
    misses = len(run.query_results) - hits
    console.print(f"\n  Hits: {hits}  Misses: {misses}")

    if misses > 0:
        console.print("\n[bold red]Missed Queries:[/bold red]")
        for qr in run.query_results:
            if not qr.hit:
                console.print(f"  - {qr.query.query}")


def report_comparison(comparison: dict[str, Any], console: Console | None = None) -> None:
    """Print side-by-side comparison of two runs.

    Args:
        comparison: Output from compare_runs().
        console: Rich console.
    """
    if console is None:
        console = Console()

    run_a = comparison["run_a_id"]
    run_b = comparison["run_b_id"]
    console.print(f"\n[bold]Comparison:[/bold] {run_a} vs {run_b}")

    table = Table(title="Metric Deltas", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Delta", justify="right")
    table.add_column("% Change", justify="right")

    for metric, delta in sorted(comparison["metric_deltas"].items()):
        pct = comparison["metric_pct_deltas"].get(metric)
        pct_str = f"{pct:+.1f}%" if pct is not None else "N/A"
        color = "green" if delta >= 0 else "red"
        table.add_row(metric, f"[{color}]{delta:+.4f}[/{color}]", pct_str)

    console.print(table)

    summary = comparison["summary"]
    console.print(
        f"\n  Wins: {summary['wins']}  Losses: {summary['losses']}  Ties: {summary['ties']}"
    )


def report_drift(drift_results: list[DriftResult], console: Console | None = None) -> None:
    """Print drift detection results with severity coloring.

    Args:
        drift_results: List of per-metric drift results.
        console: Rich console.
    """
    if console is None:
        console = Console()

    table = Table(title="Drift Detection", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Baseline", justify="right")
    table.add_column("Current", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("p-value", justify="right")
    table.add_column("Severity", justify="center")

    severity_colors = {
        DriftSeverity.INFO: "green",
        DriftSeverity.WARNING: "yellow",
        DriftSeverity.CRITICAL: "red",
    }

    for dr in drift_results:
        color = severity_colors[dr.severity]
        p_str = f"{dr.p_value:.4f}" if dr.p_value is not None else "-"
        table.add_row(
            dr.metric_name,
            f"{dr.baseline_value:.4f}",
            f"{dr.current_value:.4f}",
            f"[{color}]{dr.delta:+.4f}[/{color}]",
            p_str,
            f"[{color}]{dr.severity.value.upper()}[/{color}]",
        )

    console.print(table)
