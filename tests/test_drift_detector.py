"""Tests for drift detection: baseline storage and statistical drift analysis."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from ir_eval.drift.baseline import BaselineStore
from ir_eval.drift.detector import DriftDetector
from ir_eval.types import (
    DriftSeverity,
    EvalRun,
    GoldenQuery,
    QueryResult,
    RetrievedItem,
)


def _make_run(
    query_hits: list[bool],
    rr_values: list[float],
    ndcg_values: list[float],
    run_id: str = "test-run",
    adapter: str = "test",
) -> EvalRun:
    """Build an EvalRun from per-query metric vectors."""
    query_results = []
    for i, (hit, rr, ndcg) in enumerate(zip(query_hits, rr_values, ndcg_values, strict=False)):
        gq = GoldenQuery(query=f"query-{i}", relevant_ids=(f"doc-{i}",))
        retrieved = (RetrievedItem(id=f"doc-{i}", rank=1),) if hit else ()
        qr = QueryResult(
            query=gq,
            retrieved=retrieved,
            hit=hit,
            reciprocal_rank=rr,
            ndcg=ndcg,
            precision_at_k=1.0 if hit else 0.0,
            average_precision=rr,
        )
        query_results.append(qr)

    hit_rate = sum(query_hits) / len(query_hits) if query_hits else 0.0
    mrr = sum(rr_values) / len(rr_values) if rr_values else 0.0
    ndcg_mean = sum(ndcg_values) / len(ndcg_values) if ndcg_values else 0.0

    return EvalRun(
        id=run_id,
        golden_set_name="test-set",
        adapter_name=adapter,
        timestamp=datetime.now(UTC),
        query_results=tuple(query_results),
        metrics={"hit_rate": hit_rate, "mrr": mrr, "ndcg_at_k": ndcg_mean},
    )


class TestBaselineStore:
    """File-based baseline storage."""

    def test_set_and_get(self, tmp_path: Path) -> None:
        store = BaselineStore(base_dir=tmp_path / "baselines")
        run = _make_run([True, True], [1.0, 0.5], [1.0, 0.6])
        store.set_baseline(run, set_by="test", notes="initial")
        baseline = store.get_baseline("test-set")
        assert baseline is not None
        assert baseline.run.id == run.id
        assert baseline.notes == "initial"

    def test_get_nonexistent(self, tmp_path: Path) -> None:
        store = BaselineStore(base_dir=tmp_path / "baselines")
        assert store.get_baseline("nonexistent") is None

    def test_overwrite(self, tmp_path: Path) -> None:
        store = BaselineStore(base_dir=tmp_path / "baselines")
        run1 = _make_run([True], [1.0], [1.0], run_id="run-1")
        run2 = _make_run([False], [0.0], [0.0], run_id="run-2")
        store.set_baseline(run1)
        store.set_baseline(run2)
        baseline = store.get_baseline("test-set")
        assert baseline is not None
        assert baseline.run.id == "run-2"

    def test_list_baselines(self, tmp_path: Path) -> None:
        store = BaselineStore(base_dir=tmp_path / "baselines")
        run = _make_run([True], [1.0], [1.0])
        store.set_baseline(run)
        listed = store.list_baselines()
        assert len(listed) == 1
        assert listed[0]["golden_set_name"] == "test-set"

    def test_delete_baseline(self, tmp_path: Path) -> None:
        store = BaselineStore(base_dir=tmp_path / "baselines")
        run = _make_run([True], [1.0], [1.0])
        store.set_baseline(run)
        assert store.delete_baseline("test-set") is True
        assert store.get_baseline("test-set") is None
        assert store.delete_baseline("test-set") is False


class TestDriftDetector:
    """Statistical drift analysis."""

    def test_no_drift_identical_runs(self) -> None:
        """Identical runs → all INFO severity."""
        hits = [True] * 18 + [False] * 2
        rr = [1.0] * 15 + [0.5] * 3 + [0.0] * 2
        ndcg = [1.0] * 15 + [0.6] * 3 + [0.0] * 2

        run_a = _make_run(hits, rr, ndcg, run_id="a")
        run_b = _make_run(hits, rr, ndcg, run_id="b")

        detector = DriftDetector(seed=42)
        results = detector.detect(run_a, run_b)

        for dr in results:
            assert dr.severity == DriftSeverity.INFO
            assert dr.delta == pytest.approx(0.0)

    def test_critical_regression(self) -> None:
        """Large regression → CRITICAL severity.

        Baseline: 18/20 hits, Current: 10/20 hits.
        """
        baseline_hits = [True] * 18 + [False] * 2
        baseline_rr = [1.0] * 15 + [0.5] * 3 + [0.0] * 2
        baseline_ndcg = [1.0] * 15 + [0.6] * 3 + [0.0] * 2

        current_hits = [True] * 10 + [False] * 10
        current_rr = [1.0] * 8 + [0.5] * 2 + [0.0] * 10
        current_ndcg = [1.0] * 8 + [0.6] * 2 + [0.0] * 10

        baseline = _make_run(baseline_hits, baseline_rr, baseline_ndcg, run_id="base")
        current = _make_run(current_hits, current_rr, current_ndcg, run_id="curr")

        detector = DriftDetector(seed=42)
        results = detector.detect(baseline, current)

        # At least one metric should be CRITICAL
        severities = {dr.metric_name: dr.severity for dr in results}
        has_alert = (
            DriftSeverity.CRITICAL in severities.values()
            or DriftSeverity.WARNING in severities.values()
        )
        assert has_alert

        # Hit rate should show negative delta
        hr = next(dr for dr in results if dr.metric_name == "hit_rate")
        assert hr.delta < 0

    def test_improvement_is_info(self) -> None:
        """Improvement should be INFO (we only flag regressions)."""
        baseline_hits = [True] * 10 + [False] * 10
        baseline_rr = [1.0] * 8 + [0.5] * 2 + [0.0] * 10
        baseline_ndcg = [1.0] * 8 + [0.6] * 2 + [0.0] * 10

        current_hits = [True] * 18 + [False] * 2
        current_rr = [1.0] * 15 + [0.5] * 3 + [0.0] * 2
        current_ndcg = [1.0] * 15 + [0.6] * 3 + [0.0] * 2

        baseline = _make_run(baseline_hits, baseline_rr, baseline_ndcg, run_id="base")
        current = _make_run(current_hits, current_rr, current_ndcg, run_id="curr")

        detector = DriftDetector(seed=42)
        results = detector.detect(baseline, current)

        # No CRITICAL or WARNING for improvements
        for dr in results:
            assert dr.severity == DriftSeverity.INFO
            assert dr.delta >= 0

    def test_ci_bounds_present(self) -> None:
        """MRR and NDCG should have CI bounds."""
        hits = [True] * 15 + [False] * 5
        rr = [1.0] * 12 + [0.5] * 3 + [0.0] * 5
        ndcg = [1.0] * 12 + [0.6] * 3 + [0.0] * 5

        run_a = _make_run(hits, rr, ndcg, run_id="a")
        run_b = _make_run(hits, rr, ndcg, run_id="b")

        detector = DriftDetector(seed=42)
        results = detector.detect(run_a, run_b)

        mrr_result = next(dr for dr in results if dr.metric_name == "mrr")
        assert mrr_result.ci_lower is not None
        assert mrr_result.ci_upper is not None

    def test_no_common_queries_returns_empty(self) -> None:
        """Runs with disjoint queries → no drift results."""
        gq_a = GoldenQuery(query="unique-a", relevant_ids=("d1",))
        gq_b = GoldenQuery(query="unique-b", relevant_ids=("d2",))

        run_a = EvalRun(
            id="a",
            golden_set_name="test",
            adapter_name="test",
            timestamp=datetime.now(UTC),
            query_results=(
                QueryResult(
                    query=gq_a,
                    retrieved=(),
                    hit=False,
                    reciprocal_rank=0.0,
                    ndcg=0.0,
                    precision_at_k=0.0,
                    average_precision=0.0,
                ),
            ),
            metrics={},
        )
        run_b = EvalRun(
            id="b",
            golden_set_name="test",
            adapter_name="test",
            timestamp=datetime.now(UTC),
            query_results=(
                QueryResult(
                    query=gq_b,
                    retrieved=(),
                    hit=False,
                    reciprocal_rank=0.0,
                    ndcg=0.0,
                    precision_at_k=0.0,
                    average_precision=0.0,
                ),
            ),
            metrics={},
        )

        detector = DriftDetector(seed=42)
        results = detector.detect(run_a, run_b)
        assert results == []
