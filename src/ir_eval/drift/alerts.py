"""Alert sinks for drift detection results.

Provides configurable alert handling:
    - StdoutAlert: Print severity-colored messages
    - ExitCodeAlert: Exit with code 1 on CRITICAL (for CI)

Usage:
    alerts = [StdoutAlert(), ExitCodeAlert()]
    for alert in alerts:
        alert.process(drift_results)
"""

from __future__ import annotations

import sys
from typing import Protocol

from ir_eval.types import DriftResult, DriftSeverity


class AlertSink(Protocol):
    """Protocol for drift alert handlers."""

    def process(self, drift_results: list[DriftResult]) -> None:
        """Process drift results and take appropriate action.

        Args:
            drift_results: Per-metric drift analysis results.
        """
        ...


class StdoutAlert:
    """Print drift alerts to stdout."""

    def process(self, drift_results: list[DriftResult]) -> None:
        """Print each drift result with severity indicator.

        Args:
            drift_results: Per-metric drift analysis results.
        """
        for dr in drift_results:
            if dr.severity == DriftSeverity.CRITICAL:
                prefix = "CRITICAL"
            elif dr.severity == DriftSeverity.WARNING:
                prefix = "WARNING"
            else:
                prefix = "OK"

            p_str = f" (p={dr.p_value:.4f})" if dr.p_value is not None else ""
            print(f"[{prefix}] {dr.metric_name}: {dr.delta:+.4f} ({dr.delta_pct:+.1f}%){p_str}")


class ExitCodeAlert:
    """Exit with code 1 if any CRITICAL drift is detected.

    Designed for CI pipeline integration:
        ir-eval drift --exit-code → exits 1 on critical regression.
    """

    def process(self, drift_results: list[DriftResult]) -> None:
        """Exit with code 1 if any metric has CRITICAL severity.

        Args:
            drift_results: Per-metric drift analysis results.
        """
        has_critical = any(dr.severity == DriftSeverity.CRITICAL for dr in drift_results)
        if has_critical:
            sys.exit(1)
