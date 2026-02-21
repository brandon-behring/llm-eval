"""Tests for reporters: console, markdown, JSON."""

from __future__ import annotations

import json

from ir_eval.compare import compare_runs
from ir_eval.reporters import console as console_reporter
from ir_eval.reporters import json_reporter
from ir_eval.reporters import markdown as md_reporter
from ir_eval.types import DriftResult, DriftSeverity, EvalRun


class TestMarkdownReporter:
    """Markdown report generation."""

    def test_eval_run_report(self, sample_eval_run: EvalRun) -> None:
        md = md_reporter.report_eval_run(sample_eval_run)
        assert "# Evaluation Report" in md
        assert "stub" in md
        assert "hit_rate" in md
        assert "1.0000" in md

    def test_comparison_report(self, sample_eval_run: EvalRun) -> None:
        comparison = compare_runs(sample_eval_run, sample_eval_run)
        md = md_reporter.report_comparison(comparison)
        assert "# Comparison" in md
        assert "0 wins" in md

    def test_drift_report(self) -> None:
        drift_results = [
            DriftResult(
                metric_name="hit_rate",
                baseline_value=0.93,
                current_value=0.85,
                delta=-0.08,
                delta_pct=-8.6,
                p_value=0.03,
                severity=DriftSeverity.CRITICAL,
            ),
            DriftResult(
                metric_name="mrr",
                baseline_value=0.85,
                current_value=0.84,
                delta=-0.01,
                delta_pct=-1.2,
                severity=DriftSeverity.INFO,
            ),
        ]
        md = md_reporter.report_drift(drift_results)
        assert "# Drift Detection Report" in md
        assert "CRIT" in md
        assert "hit_rate" in md


class TestJsonReporter:
    """JSON report generation."""

    def test_eval_run_is_valid_json(self, sample_eval_run: EvalRun) -> None:
        result = json_reporter.report_eval_run(sample_eval_run)
        data = json.loads(result)
        assert data["id"] == sample_eval_run.id

    def test_comparison_is_valid_json(self, sample_eval_run: EvalRun) -> None:
        comparison = compare_runs(sample_eval_run, sample_eval_run)
        result = json_reporter.report_comparison(comparison)
        data = json.loads(result)
        assert "metric_deltas" in data

    def test_drift_is_valid_json(self) -> None:
        drift_results = [
            DriftResult(
                metric_name="hit_rate",
                baseline_value=0.93,
                current_value=0.85,
                delta=-0.08,
                delta_pct=-8.6,
                severity=DriftSeverity.WARNING,
            ),
        ]
        result = json_reporter.report_drift(drift_results)
        data = json.loads(result)
        assert data["has_warning"] is True
        assert data["has_critical"] is False

    def test_drift_critical_flag(self) -> None:
        drift_results = [
            DriftResult(
                metric_name="hit_rate",
                baseline_value=0.93,
                current_value=0.80,
                delta=-0.13,
                delta_pct=-14.0,
                severity=DriftSeverity.CRITICAL,
            ),
        ]
        result = json_reporter.report_drift(drift_results)
        data = json.loads(result)
        assert data["has_critical"] is True
        assert data["overall_severity"] == "critical"


class TestConsoleReporter:
    """Console reporter — tests it runs without error (not output matching)."""

    def test_eval_run_no_crash(self, sample_eval_run: EvalRun) -> None:
        """Just ensure it doesn't raise."""
        from rich.console import Console

        c = Console(file=open("/dev/null", "w"))
        console_reporter.report_eval_run(sample_eval_run, console=c)

    def test_drift_no_crash(self) -> None:
        from rich.console import Console

        drift_results = [
            DriftResult(
                metric_name="hit_rate",
                baseline_value=0.93,
                current_value=0.85,
                delta=-0.08,
                delta_pct=-8.6,
                severity=DriftSeverity.WARNING,
            ),
        ]
        c = Console(file=open("/dev/null", "w"))
        console_reporter.report_drift(drift_results, console=c)
