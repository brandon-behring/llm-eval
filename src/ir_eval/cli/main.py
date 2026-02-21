"""CLI for ir-eval.

Commands:
    ir-eval run <golden-set> --adapter <name> [--top-k 10]
    ir-eval baseline set <run-path> [--notes "..."]
    ir-eval baseline show <golden-set-name>
    ir-eval compare <run-a> <run-b> [--format console|markdown|json]
    ir-eval drift <golden-set> --adapter <name> [--ci] [--exit-code]
    ir-eval validate <golden-set>
    ir-eval history <golden-set-name>
"""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from typing import Annotated

import typer

from ir_eval.compare import compare_runs
from ir_eval.drift.baseline import BaselineStore
from ir_eval.drift.detector import DriftDetector
from ir_eval.reporters import console as console_reporter
from ir_eval.reporters import json_reporter
from ir_eval.reporters import markdown as md_reporter
from ir_eval.types import EvalRun, GoldenSet, ResultSet

app = typer.Typer(
    name="ir-eval",
    help="Statistical RAG evaluation framework with drift detection.",
    no_args_is_help=True,
)

baseline_app = typer.Typer(help="Manage evaluation baselines.")
app.add_typer(baseline_app, name="baseline")


def _load_adapter(name: str):  # type: ignore[no-untyped-def]
    """Load adapter by entry point name.

    Args:
        name: Adapter name registered as an entry point.

    Returns:
        Instantiated adapter.

    Raises:
        typer.Exit: If adapter not found.
    """
    try:
        eps = importlib.metadata.entry_points(group="ir_eval.adapters")
        for ep in eps:
            if ep.name == name:
                adapter_cls = ep.load()
                return adapter_cls()
        typer.echo(f"Error: Adapter '{name}' not found.", err=True)
        typer.echo("Available adapters:", err=True)
        for ep in eps:
            typer.echo(f"  - {ep.name}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error loading adapter '{name}': {e}", err=True)
        raise typer.Exit(1) from None


@app.command()
def evaluate(
    results_path: Annotated[Path, typer.Argument(help="Path to results JSON file")],
    golden: Annotated[Path, typer.Option("--golden", help="Path to golden set file (JSON/YAML)")],
    top_k: Annotated[int, typer.Option(help="Cutoff for metrics")] = 10,
    output: Annotated[Path | None, typer.Option(help="Save run to JSON")] = None,
    format: Annotated[str, typer.Option(help="Output format")] = "console",
) -> None:
    """Evaluate pre-computed retrieval results against a golden set.

    This is the primary evaluation path — no live adapter needed.
    Export your retrieval system's results to JSON, then evaluate::

        ir-eval evaluate results.json --golden golden.json
    """
    from ir_eval.runner import evaluate_from_results

    golden_set = GoldenSet.from_file(golden)
    result_set = ResultSet.from_json(results_path)
    eval_run = evaluate_from_results(golden_set, result_set, top_k=top_k)

    if format == "json":
        typer.echo(json_reporter.report_eval_run(eval_run))
    elif format == "markdown":
        typer.echo(md_reporter.report_eval_run(eval_run))
    else:
        console_reporter.report_eval_run(eval_run)

    if output:
        eval_run.to_json(output)
        typer.echo(f"\nRun saved to: {output}")


@app.command()
def run(
    golden_set_path: Annotated[Path, typer.Argument(help="Path to golden set file (JSON/YAML)")],
    adapter: Annotated[str, typer.Option(help="Adapter name (entry point)")],
    top_k: Annotated[int, typer.Option(help="Number of results to retrieve")] = 10,
    output: Annotated[Path | None, typer.Option(help="Save run to JSON")] = None,
    format: Annotated[str, typer.Option(help="Output format")] = "console",
) -> None:
    """Run evaluation of a retrieval adapter against a golden set."""
    from ir_eval.runner import run_evaluation

    golden = GoldenSet.from_file(golden_set_path)
    retrieval_adapter = _load_adapter(adapter)
    eval_run = run_evaluation(golden, retrieval_adapter, top_k=top_k)

    if format == "json":
        typer.echo(json_reporter.report_eval_run(eval_run))
    elif format == "markdown":
        typer.echo(md_reporter.report_eval_run(eval_run))
    else:
        console_reporter.report_eval_run(eval_run)

    if output:
        eval_run.to_json(output)
        typer.echo(f"\nRun saved to: {output}")


@baseline_app.command("set")
def baseline_set(
    run_path: Annotated[Path, typer.Argument(help="Path to evaluation run JSON")],
    notes: Annotated[str, typer.Option(help="Notes for this baseline")] = "",
    store_dir: Annotated[Path | None, typer.Option(help="Baseline store directory")] = None,
) -> None:
    """Set a run as the baseline for its golden set."""
    eval_run = EvalRun.from_json(run_path)
    store = BaselineStore(base_dir=store_dir)
    store.set_baseline(eval_run, set_by="cli", notes=notes)
    typer.echo(f"Baseline set for '{eval_run.golden_set_name}'")
    typer.echo(f"  Run: {eval_run.id}")
    typer.echo(f"  Metrics: {json.dumps(eval_run.metrics, indent=2)}")


@baseline_app.command("show")
def baseline_show(
    golden_set_name: Annotated[str, typer.Argument(help="Golden set name")],
    store_dir: Annotated[Path | None, typer.Option(help="Baseline store directory")] = None,
) -> None:
    """Show the current baseline for a golden set."""
    store = BaselineStore(base_dir=store_dir)
    baseline = store.get_baseline(golden_set_name)
    if baseline is None:
        typer.echo(f"No baseline found for '{golden_set_name}'")
        raise typer.Exit(1)

    typer.echo(f"Baseline for '{golden_set_name}':")
    typer.echo(f"  Run ID: {baseline.run.id}")
    typer.echo(f"  Adapter: {baseline.run.adapter_name}")
    typer.echo(f"  Set at: {baseline.set_at.isoformat()}")
    typer.echo(f"  Notes: {baseline.notes}")
    typer.echo(f"  Metrics: {json.dumps(baseline.run.metrics, indent=2)}")


@app.command()
def compare(
    run_a_path: Annotated[Path, typer.Argument(help="Path to first run JSON")],
    run_b_path: Annotated[Path, typer.Argument(help="Path to second run JSON")],
    format: Annotated[str, typer.Option(help="Output format")] = "console",
) -> None:
    """Compare two evaluation runs side-by-side."""
    run_a = EvalRun.from_json(run_a_path)
    run_b = EvalRun.from_json(run_b_path)
    comparison = compare_runs(run_a, run_b)

    if format == "json":
        typer.echo(json_reporter.report_comparison(comparison))
    elif format == "markdown":
        typer.echo(md_reporter.report_comparison(comparison))
    else:
        console_reporter.report_comparison(comparison)


@app.command()
def drift(
    golden_set_path: Annotated[Path, typer.Argument(help="Path to golden set file")],
    adapter: Annotated[str, typer.Option(help="Adapter name")],
    top_k: Annotated[int, typer.Option(help="Top-k for retrieval")] = 10,
    exit_code: Annotated[bool, typer.Option(help="Exit 1 on critical drift")] = False,
    ci: Annotated[bool, typer.Option(help="Show confidence intervals")] = False,
    format: Annotated[str, typer.Option(help="Output format")] = "console",
    store_dir: Annotated[Path | None, typer.Option(help="Baseline store directory")] = None,
) -> None:
    """Run evaluation and compare against baseline for drift detection."""
    from ir_eval.drift.alerts import ExitCodeAlert
    from ir_eval.runner import run_evaluation

    golden = GoldenSet.from_file(golden_set_path)
    store = BaselineStore(base_dir=store_dir)
    baseline = store.get_baseline(golden.name)

    if baseline is None:
        typer.echo(f"No baseline found for '{golden.name}'. Run 'ir-eval baseline set' first.")
        raise typer.Exit(1)

    retrieval_adapter = _load_adapter(adapter)
    current_run = run_evaluation(golden, retrieval_adapter, top_k=top_k)

    detector = DriftDetector(seed=42)
    results = detector.detect(baseline.run, current_run)

    if format == "json":
        typer.echo(json_reporter.report_drift(results))
    elif format == "markdown":
        typer.echo(md_reporter.report_drift(results))
    else:
        console_reporter.report_drift(results)

    if exit_code:
        from ir_eval.drift.alerts import ExitCodeAlert

        ExitCodeAlert().process(results)


@app.command()
def validate(
    golden_set_path: Annotated[Path, typer.Argument(help="Path to golden set file")],
) -> None:
    """Validate a golden set file structure."""
    try:
        golden = GoldenSet.from_file(golden_set_path)
        typer.echo(f"Valid golden set: '{golden.name}' v{golden.version}")
        typer.echo(f"  Queries: {len(golden.queries)}")

        # Domain distribution
        domains: dict[str, int] = {}
        difficulties: dict[str, int] = {}
        for q in golden.queries:
            domains[q.domain or "none"] = domains.get(q.domain or "none", 0) + 1
            difficulties[q.difficulty.value] = difficulties.get(q.difficulty.value, 0) + 1

        typer.echo(f"  Domains: {domains}")
        typer.echo(f"  Difficulties: {difficulties}")

        graded = sum(1 for q in golden.queries if q.relevance_grades)
        typer.echo(f"  Graded relevance: {graded}/{len(golden.queries)}")

    except Exception as e:
        typer.echo(f"Invalid golden set: {e}", err=True)
        raise typer.Exit(1) from None


@app.command()
def history(
    golden_set_name: Annotated[str, typer.Argument(help="Golden set name")],
    store_dir: Annotated[Path | None, typer.Option(help="Baseline store directory")] = None,
) -> None:
    """Show baseline history for a golden set."""
    store = BaselineStore(base_dir=store_dir)
    baselines = store.list_baselines()

    matching = [b for b in baselines if b["golden_set_name"] == golden_set_name]
    if not matching:
        typer.echo(f"No baselines found for '{golden_set_name}'")
        raise typer.Exit(0)

    for b in matching:
        typer.echo(f"  {b['timestamp']}  {b['adapter_name']}  {b['notes']}")
