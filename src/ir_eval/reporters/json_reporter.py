"""JSON reporter for CI pipeline integration.

Produces machine-readable JSON for automated quality gates.
"""

from __future__ import annotations

import json
from typing import Any

from ir_eval.types import DriftResult, DriftSeverity, EvalRun


def report_eval_run(run: EvalRun) -> str:
    """Generate JSON report for an evaluation run.

    Args:
        run: The evaluation run.

    Returns:
        JSON string.
    """
    return json.dumps(run.to_dict(), indent=2, default=str)


def report_comparison(comparison: dict[str, Any]) -> str:
    """Generate JSON comparison report.

    Args:
        comparison: Output from compare_runs().

    Returns:
        JSON string.
    """
    return json.dumps(comparison, indent=2, default=str)


def report_drift(drift_results: list[DriftResult]) -> str:
    """Generate JSON drift report.

    Args:
        drift_results: Per-metric drift results.

    Returns:
        JSON string with results and overall status.
    """
    max_severity = max(
        (dr.severity for dr in drift_results),
        default=DriftSeverity.INFO,
        key=lambda s: list(DriftSeverity).index(s),
    )

    report = {
        "overall_severity": max_severity.value,
        "has_critical": any(dr.severity == DriftSeverity.CRITICAL for dr in drift_results),
        "has_warning": any(dr.severity == DriftSeverity.WARNING for dr in drift_results),
        "results": [dr.to_dict() for dr in drift_results],
    }

    return json.dumps(report, indent=2)
