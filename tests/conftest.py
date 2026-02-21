"""Shared test fixtures for ir-eval."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from ir_eval.types import (
    Difficulty,
    EvalRun,
    GoldenQuery,
    GoldenSet,
    QueryResult,
    RetrievedItem,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def golden_small_path() -> Path:
    """Path to the 10-query test fixture."""
    return FIXTURES_DIR / "golden_small.json"


@pytest.fixture
def golden_research_kb_path() -> Path:
    """Path to the 47-query research-kb fixture."""
    return FIXTURES_DIR / "golden_research_kb.json"


@pytest.fixture
def golden_small(golden_small_path: Path) -> GoldenSet:
    """Load the 10-query golden set."""
    return GoldenSet.from_json(golden_small_path)


@pytest.fixture
def sample_query() -> GoldenQuery:
    """A single sample golden query."""
    return GoldenQuery(
        query="d-separation",
        relevant_ids=("doc-a", "doc-b"),
        domain="causal_inference",
        difficulty=Difficulty.EASY,
    )


@pytest.fixture
def sample_retrieved() -> list[RetrievedItem]:
    """Sample retrieval results: doc-b at rank 2, doc-a at rank 4."""
    return [
        RetrievedItem(id="doc-x", rank=1, score=0.95),
        RetrievedItem(id="doc-b", rank=2, score=0.90),
        RetrievedItem(id="doc-y", rank=3, score=0.85),
        RetrievedItem(id="doc-a", rank=4, score=0.80),
        RetrievedItem(id="doc-z", rank=5, score=0.75),
    ]


@pytest.fixture
def sample_eval_run() -> EvalRun:
    """A minimal EvalRun for testing."""
    query = GoldenQuery(
        query="test query",
        relevant_ids=("doc-a",),
    )
    qr = QueryResult(
        query=query,
        retrieved=(RetrievedItem(id="doc-a", rank=1, score=0.9),),
        hit=True,
        reciprocal_rank=1.0,
        ndcg=1.0,
        precision_at_k=1.0,
        average_precision=1.0,
    )
    return EvalRun(
        id="test-run-001",
        golden_set_name="test-set",
        adapter_name="stub",
        timestamp=datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC),
        query_results=(qr,),
        metrics={"hit_rate": 1.0, "mrr": 1.0, "ndcg_at_5": 1.0},
        config={"top_k": 10},
    )
