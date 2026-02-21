"""Tests for core types: GoldenQuery, GoldenSet, EvalRun, etc."""

from __future__ import annotations

from pathlib import Path

import pytest

from ir_eval.types import (
    Baseline,
    Difficulty,
    DriftResult,
    DriftSeverity,
    EvalRun,
    GoldenQuery,
    GoldenSet,
    QueryResult,
    RetrievedItem,
)


class TestGoldenQuery:
    """GoldenQuery construction and serialization."""

    def test_basic_construction(self) -> None:
        q = GoldenQuery(query="test", relevant_ids=("a", "b"))
        assert q.query == "test"
        assert q.relevant_ids == ("a", "b")
        assert q.domain is None
        assert q.difficulty == Difficulty.EASY

    def test_empty_query_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            GoldenQuery(query="", relevant_ids=("a",))

    def test_whitespace_query_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            GoldenQuery(query="   ", relevant_ids=("a",))

    def test_empty_relevant_ids_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            GoldenQuery(query="test", relevant_ids=())

    def test_roundtrip_dict(self) -> None:
        q = GoldenQuery(
            query="test query",
            relevant_ids=("a", "b"),
            domain="causal_inference",
            difficulty=Difficulty.MEDIUM,
            relevance_grades={"a": 3, "b": 1},
            metadata={"source": "unit-test"},
        )
        d = q.to_dict()
        q2 = GoldenQuery.from_dict(d)
        assert q2.query == q.query
        assert q2.relevant_ids == q.relevant_ids
        assert q2.domain == q.domain
        assert q2.difficulty == q.difficulty
        assert q2.relevance_grades == q.relevance_grades
        assert q2.metadata == q.metadata

    def test_minimal_dict_serialization(self) -> None:
        """Defaults should not appear in serialized dict."""
        q = GoldenQuery(query="test", relevant_ids=("a",))
        d = q.to_dict()
        assert "domain" not in d
        assert "difficulty" not in d
        assert "relevance_grades" not in d
        assert "metadata" not in d


class TestGoldenSet:
    """GoldenSet construction, serialization, and filtering."""

    def test_basic_construction(self) -> None:
        q = GoldenQuery(query="q1", relevant_ids=("a",))
        gs = GoldenSet(name="test", version="1.0", queries=(q,))
        assert gs.name == "test"
        assert len(gs.queries) == 1

    def test_empty_name_raises(self) -> None:
        q = GoldenQuery(query="q1", relevant_ids=("a",))
        with pytest.raises(ValueError, match="non-empty"):
            GoldenSet(name="", version="1.0", queries=(q,))

    def test_empty_queries_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            GoldenSet(name="test", version="1.0", queries=())

    def test_json_roundtrip(self, tmp_path: Path) -> None:
        q = GoldenQuery(
            query="test query",
            relevant_ids=("a", "b"),
            domain="time_series",
        )
        gs = GoldenSet(name="test-set", version="1.0.0", queries=(q,), description="Test")
        path = tmp_path / "golden.json"
        gs.to_json(path)
        gs2 = GoldenSet.from_json(path)
        assert gs2.name == gs.name
        assert gs2.version == gs.version
        assert len(gs2.queries) == 1
        assert gs2.queries[0].query == "test query"
        assert gs2.description == "Test"

    def test_yaml_roundtrip(self, tmp_path: Path) -> None:
        q = GoldenQuery(query="q1", relevant_ids=("a",), domain="ci")
        gs = GoldenSet(name="yaml-test", version="2.0", queries=(q,))
        path = tmp_path / "golden.yaml"
        gs.to_yaml(path)
        gs2 = GoldenSet.from_yaml(path)
        assert gs2.name == "yaml-test"
        assert gs2.queries[0].domain == "ci"

    def test_from_file_json(self, golden_small_path: Path) -> None:
        gs = GoldenSet.from_file(golden_small_path)
        assert gs.name == "small-test-set"
        assert len(gs.queries) == 10

    def test_from_file_yaml(self, tmp_path: Path) -> None:
        q = GoldenQuery(query="q1", relevant_ids=("a",))
        gs = GoldenSet(name="t", version="1.0", queries=(q,))
        path = tmp_path / "golden.yml"
        gs.to_yaml(path)
        gs2 = GoldenSet.from_file(path)
        assert gs2.name == "t"

    def test_filter_by_domain(self, golden_small: GoldenSet) -> None:
        filtered = golden_small.filter_by_domain("causal_inference")
        assert all(q.domain == "causal_inference" for q in filtered.queries)
        assert len(filtered.queries) >= 1

    def test_filter_by_domain_no_match(self, golden_small: GoldenSet) -> None:
        with pytest.raises(ValueError, match="No queries"):
            golden_small.filter_by_domain("nonexistent_domain")

    def test_filter_by_difficulty(self, golden_small: GoldenSet) -> None:
        filtered = golden_small.filter_by_difficulty(Difficulty.MEDIUM)
        assert all(q.difficulty == Difficulty.MEDIUM for q in filtered.queries)

    def test_load_research_kb_fixture(self, golden_research_kb_path: Path) -> None:
        gs = GoldenSet.from_json(golden_research_kb_path)
        assert gs.name == "research-kb-v1"
        assert len(gs.queries) == 47


class TestRetrievedItem:
    """RetrievedItem construction and validation."""

    def test_basic_construction(self) -> None:
        item = RetrievedItem(id="doc-1", rank=1, score=0.95)
        assert item.id == "doc-1"
        assert item.rank == 1

    def test_invalid_rank_raises(self) -> None:
        with pytest.raises(ValueError, match="Rank must be >= 1"):
            RetrievedItem(id="doc-1", rank=0)

    def test_roundtrip_dict(self) -> None:
        item = RetrievedItem(
            id="doc-1",
            rank=3,
            score=0.8,
            content="some text",
            metadata={"source": "test"},
        )
        d = item.to_dict()
        item2 = RetrievedItem.from_dict(d)
        assert item2.id == item.id
        assert item2.rank == item.rank
        assert item2.score == item.score
        assert item2.content == item.content


class TestQueryResult:
    """QueryResult construction and serialization."""

    def test_roundtrip_dict(self, sample_query: GoldenQuery) -> None:
        qr = QueryResult(
            query=sample_query,
            retrieved=(RetrievedItem(id="a", rank=1),),
            hit=True,
            reciprocal_rank=1.0,
            ndcg=0.95,
            precision_at_k=0.5,
            average_precision=0.75,
            latency_ms=42.5,
        )
        d = qr.to_dict()
        qr2 = QueryResult.from_dict(d)
        assert qr2.hit is True
        assert qr2.reciprocal_rank == 1.0
        assert qr2.latency_ms == 42.5


class TestEvalRun:
    """EvalRun construction and serialization."""

    def test_roundtrip_dict(self, sample_eval_run: EvalRun) -> None:
        d = sample_eval_run.to_dict()
        run2 = EvalRun.from_dict(d)
        assert run2.id == sample_eval_run.id
        assert run2.metrics == sample_eval_run.metrics
        assert len(run2.query_results) == 1

    def test_json_roundtrip(self, sample_eval_run: EvalRun, tmp_path: Path) -> None:
        path = tmp_path / "run.json"
        sample_eval_run.to_json(path)
        run2 = EvalRun.from_json(path)
        assert run2.id == sample_eval_run.id
        assert run2.adapter_name == "stub"

    def test_timestamp_preserved(self, sample_eval_run: EvalRun) -> None:
        d = sample_eval_run.to_dict()
        run2 = EvalRun.from_dict(d)
        assert run2.timestamp == sample_eval_run.timestamp


class TestBaseline:
    """Baseline wrapping an EvalRun."""

    def test_roundtrip_dict(self, sample_eval_run: EvalRun) -> None:
        b = Baseline(
            run=sample_eval_run,
            set_by="test",
            notes="initial baseline",
        )
        d = b.to_dict()
        b2 = Baseline.from_dict(d)
        assert b2.run.id == sample_eval_run.id
        assert b2.set_by == "test"
        assert b2.notes == "initial baseline"


class TestDriftResult:
    """DriftResult construction."""

    def test_basic_construction(self) -> None:
        dr = DriftResult(
            metric_name="hit_rate",
            baseline_value=0.93,
            current_value=0.85,
            delta=-0.08,
            delta_pct=-8.6,
            p_value=0.03,
            ci_lower=-0.12,
            ci_upper=-0.04,
            severity=DriftSeverity.CRITICAL,
            test_name="paired_bootstrap",
        )
        assert dr.severity == DriftSeverity.CRITICAL

    def test_roundtrip_dict(self) -> None:
        dr = DriftResult(
            metric_name="mrr",
            baseline_value=0.85,
            current_value=0.82,
            delta=-0.03,
            delta_pct=-3.5,
            severity=DriftSeverity.WARNING,
        )
        d = dr.to_dict()
        dr2 = DriftResult.from_dict(d)
        assert dr2.metric_name == "mrr"
        assert dr2.severity == DriftSeverity.WARNING
