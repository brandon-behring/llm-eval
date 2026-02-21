"""Tests for runner.py — evaluation orchestrator."""

from __future__ import annotations

from ir_eval.runner import run_evaluation
from ir_eval.types import GoldenQuery, GoldenSet, RetrievedItem


class StubAdapter:
    """Test adapter that returns predetermined results."""

    def __init__(self, results_map: dict[str, list[RetrievedItem]] | None = None) -> None:
        self._results = results_map or {}

    @property
    def name(self) -> str:
        return "stub"

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedItem]:
        return self._results.get(query, [])[:top_k]


class PerfectAdapter:
    """Always returns the relevant docs at rank 1."""

    def __init__(self, golden_set: GoldenSet) -> None:
        self._golden = golden_set

    @property
    def name(self) -> str:
        return "perfect"

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedItem]:
        for gq in self._golden.queries:
            if gq.query == query:
                return [
                    RetrievedItem(id=rid, rank=i + 1, score=1.0 - i * 0.1)
                    for i, rid in enumerate(gq.relevant_ids[:top_k])
                ]
        return []


class EmptyAdapter:
    """Returns no results for any query."""

    @property
    def name(self) -> str:
        return "empty"

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedItem]:
        return []


class TestRunEvaluation:
    """run_evaluation with various adapters."""

    def test_perfect_adapter(self, golden_small: GoldenSet) -> None:
        """Perfect adapter should get 100% on everything."""
        adapter = PerfectAdapter(golden_small)
        run = run_evaluation(golden_small, adapter, top_k=10)

        assert run.metrics["hit_rate"] == 1.0
        assert run.metrics["mrr"] == 1.0
        assert run.adapter_name == "perfect"
        assert run.golden_set_name == "small-test-set"
        assert len(run.query_results) == 10

    def test_empty_adapter(self, golden_small: GoldenSet) -> None:
        """Empty adapter should get 0 on everything."""
        adapter = EmptyAdapter()
        run = run_evaluation(golden_small, adapter, top_k=10)

        assert run.metrics["hit_rate"] == 0.0
        assert run.metrics["mrr"] == 0.0
        assert run.metrics["ndcg_at_k"] == 0.0

    def test_partial_adapter(self) -> None:
        """Adapter that hits some queries but not others."""
        q1 = GoldenQuery(query="q1", relevant_ids=("a",))
        q2 = GoldenQuery(query="q2", relevant_ids=("b",))
        gs = GoldenSet(name="test", version="1.0", queries=(q1, q2))

        adapter = StubAdapter({
            "q1": [RetrievedItem(id="a", rank=1, score=0.9)],
            "q2": [RetrievedItem(id="x", rank=1, score=0.9)],  # miss
        })

        run = run_evaluation(gs, adapter, top_k=5)
        assert run.metrics["hit_rate"] == 0.5
        assert run.metrics["mrr"] == 0.5  # (1.0 + 0.0) / 2

    def test_latency_recorded(self, golden_small: GoldenSet) -> None:
        """Latency should be non-negative for all queries."""
        adapter = PerfectAdapter(golden_small)
        run = run_evaluation(golden_small, adapter, top_k=5)

        for qr in run.query_results:
            assert qr.latency_ms is not None
            assert qr.latency_ms >= 0

    def test_config_includes_top_k(self, golden_small: GoldenSet) -> None:
        adapter = PerfectAdapter(golden_small)
        run = run_evaluation(golden_small, adapter, top_k=7)
        assert run.config["top_k"] == 7

    def test_run_id_unique(self, golden_small: GoldenSet) -> None:
        adapter = PerfectAdapter(golden_small)
        run1 = run_evaluation(golden_small, adapter)
        run2 = run_evaluation(golden_small, adapter)
        assert run1.id != run2.id

    def test_json_roundtrip(self, golden_small: GoldenSet, tmp_path) -> None:
        """Run should survive JSON serialization."""
        from ir_eval.types import EvalRun

        adapter = PerfectAdapter(golden_small)
        run = run_evaluation(golden_small, adapter, top_k=5)
        path = tmp_path / "run.json"
        run.to_json(path)
        run2 = EvalRun.from_json(path)
        assert run2.metrics["hit_rate"] == run.metrics["hit_rate"]
