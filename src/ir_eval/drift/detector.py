"""Statistical drift detection between evaluation runs.

Compares a current run against a baseline using per-query metrics and
statistical tests. The key insight: per-query storage enables paired tests,
which are far more powerful than aggregate-only comparisons.

Severity classification:
    - INFO: No significant change
    - WARNING: >5% drop AND p < 0.10
    - CRITICAL: >10% drop AND p < 0.05

Usage:
    from ir_eval.drift.detector import DriftDetector

    detector = DriftDetector()
    report = detector.detect(baseline_run, current_run)
"""

from __future__ import annotations

from ir_eval.metrics.confidence import (
    bootstrap_ci,
    mcnemar_test,
    paired_bootstrap_test,
)
from ir_eval.types import DriftResult, DriftSeverity, EvalRun


class DriftDetector:
    """Detects quality regression between evaluation runs.

    Args:
        ci_confidence: Confidence level for bootstrap CIs.
        n_bootstrap: Number of bootstrap samples.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        ci_confidence: float = 0.95,
        n_bootstrap: int = 10000,
        seed: int | None = None,
    ) -> None:
        self.ci_confidence = ci_confidence
        self.n_bootstrap = n_bootstrap
        self.seed = seed

    def _classify_severity(
        self,
        delta_pct: float,
        p_value: float | None,
    ) -> DriftSeverity:
        """Classify drift severity based on magnitude and significance.

        Args:
            delta_pct: Percentage change (negative = regression).
            p_value: Statistical significance.

        Returns:
            DriftSeverity classification.
        """
        if p_value is None:
            # No test possible — use magnitude only
            if delta_pct < -10:
                return DriftSeverity.WARNING
            return DriftSeverity.INFO

        if delta_pct < -10 and p_value < 0.05:
            return DriftSeverity.CRITICAL
        if delta_pct < -5 and p_value < 0.10:
            return DriftSeverity.WARNING
        return DriftSeverity.INFO

    def detect(
        self,
        baseline_run: EvalRun,
        current_run: EvalRun,
    ) -> list[DriftResult]:
        """Compare current run against baseline and detect drift.

        Runs paired statistical tests on per-query metrics. For hit_rate,
        also runs McNemar's test (more appropriate for binary outcomes).

        Args:
            baseline_run: The reference evaluation run.
            current_run: The new evaluation run to check.

        Returns:
            List of DriftResult, one per metric.
        """
        results: list[DriftResult] = []

        # Align queries by text
        baseline_by_query = {qr.query.query: qr for qr in baseline_run.query_results}
        current_by_query = {qr.query.query: qr for qr in current_run.query_results}
        common_queries = sorted(set(baseline_by_query.keys()) & set(current_by_query.keys()))

        if not common_queries:
            return results

        # Per-query metric vectors (aligned)
        baseline_rr = [baseline_by_query[q].reciprocal_rank for q in common_queries]
        current_rr = [current_by_query[q].reciprocal_rank for q in common_queries]
        baseline_ndcg = [baseline_by_query[q].ndcg for q in common_queries]
        current_ndcg = [current_by_query[q].ndcg for q in common_queries]
        baseline_hits = [baseline_by_query[q].hit for q in common_queries]
        current_hits = [current_by_query[q].hit for q in common_queries]

        # --- Hit Rate (with McNemar) ---
        baseline_hr = sum(baseline_hits) / len(baseline_hits)
        current_hr = sum(current_hits) / len(current_hits)

        pairs = list(zip(baseline_hits, current_hits, strict=False))
        both_correct = sum(1 for b, c in pairs if b and c)
        baseline_only = sum(1 for b, c in pairs if b and not c)
        current_only = sum(1 for b, c in pairs if not b and c)
        both_wrong = sum(1 for b, c in pairs if not b and not c)

        mcnemar_result = mcnemar_test(both_correct, baseline_only, current_only, both_wrong)

        hr_delta = current_hr - baseline_hr
        hr_pct = (hr_delta / baseline_hr * 100) if baseline_hr > 0 else 0.0

        results.append(DriftResult(
            metric_name="hit_rate",
            baseline_value=baseline_hr,
            current_value=current_hr,
            delta=hr_delta,
            delta_pct=hr_pct,
            p_value=mcnemar_result.p_value,
            severity=self._classify_severity(hr_pct, mcnemar_result.p_value),
            test_name="mcnemar",
        ))

        # --- MRR (with paired bootstrap) ---
        mrr_test = paired_bootstrap_test(
            baseline_rr, current_rr,
            n_bootstrap=self.n_bootstrap, seed=self.seed,
        )
        mrr_ci = bootstrap_ci(
            [c - b for b, c in zip(baseline_rr, current_rr, strict=False)],
            n_bootstrap=self.n_bootstrap,
            confidence_level=self.ci_confidence,
            seed=self.seed,
        )

        baseline_mrr = sum(baseline_rr) / len(baseline_rr)
        current_mrr = sum(current_rr) / len(current_rr)
        mrr_delta = current_mrr - baseline_mrr
        mrr_pct = (mrr_delta / baseline_mrr * 100) if baseline_mrr > 0 else 0.0

        results.append(DriftResult(
            metric_name="mrr",
            baseline_value=baseline_mrr,
            current_value=current_mrr,
            delta=mrr_delta,
            delta_pct=mrr_pct,
            p_value=mrr_test.p_value,
            ci_lower=mrr_ci.lower,
            ci_upper=mrr_ci.upper,
            severity=self._classify_severity(mrr_pct, mrr_test.p_value),
            test_name="paired_bootstrap",
        ))

        # --- NDCG (with paired bootstrap) ---
        ndcg_test = paired_bootstrap_test(
            baseline_ndcg, current_ndcg,
            n_bootstrap=self.n_bootstrap, seed=self.seed,
        )
        ndcg_ci = bootstrap_ci(
            [c - b for b, c in zip(baseline_ndcg, current_ndcg, strict=False)],
            n_bootstrap=self.n_bootstrap,
            confidence_level=self.ci_confidence,
            seed=self.seed,
        )

        baseline_ndcg_mean = sum(baseline_ndcg) / len(baseline_ndcg)
        current_ndcg_mean = sum(current_ndcg) / len(current_ndcg)
        ndcg_delta = current_ndcg_mean - baseline_ndcg_mean
        ndcg_pct = (ndcg_delta / baseline_ndcg_mean * 100) if baseline_ndcg_mean > 0 else 0.0

        results.append(DriftResult(
            metric_name="ndcg",
            baseline_value=baseline_ndcg_mean,
            current_value=current_ndcg_mean,
            delta=ndcg_delta,
            delta_pct=ndcg_pct,
            p_value=ndcg_test.p_value,
            ci_lower=ndcg_ci.lower,
            ci_upper=ndcg_ci.upper,
            severity=self._classify_severity(ndcg_pct, ndcg_test.p_value),
            test_name="paired_bootstrap",
        ))

        return results
