"""Core types for the ir-eval framework.

Defines the data model: golden sets, retrieval results, evaluation runs,
baselines, and drift results. All types are immutable dataclasses with
JSON/YAML serialization.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml


class Difficulty(StrEnum):
    """Query difficulty level for stratified analysis."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class DriftSeverity(StrEnum):
    """Severity of detected metric drift."""

    INFO = "info"  # No significant change
    WARNING = "warning"  # >5% drop AND p<0.10
    CRITICAL = "critical"  # >10% drop AND p<0.05


# ---------------------------------------------------------------------------
# Golden set types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GoldenQuery:
    """A single query with known relevant documents.

    Args:
        query: The search query text.
        relevant_ids: IDs of documents known to be relevant.
        domain: Optional knowledge domain (e.g., "causal_inference").
        difficulty: Query difficulty for stratified analysis.
        relevance_grades: Optional mapping of doc_id -> relevance grade (0-3).
            If provided, enables graded metrics like NDCG. If absent,
            all relevant_ids are treated as binary relevant (grade=1).
        metadata: Optional extra fields (source_title, notes, etc.).
    """

    query: str
    relevant_ids: tuple[str, ...]
    domain: str | None = None
    difficulty: Difficulty = Difficulty.EASY
    relevance_grades: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("Query must be non-empty")
        if not self.relevant_ids:
            raise ValueError(f"Query '{self.query}' must have at least one relevant_id")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        d: dict[str, Any] = {
            "query": self.query,
            "relevant_ids": list(self.relevant_ids),
        }
        if self.domain is not None:
            d["domain"] = self.domain
        if self.difficulty != Difficulty.EASY:
            d["difficulty"] = self.difficulty.value
        if self.relevance_grades:
            d["relevance_grades"] = self.relevance_grades
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GoldenQuery:
        """Deserialize from dict."""
        return cls(
            query=d["query"],
            relevant_ids=tuple(d["relevant_ids"]),
            domain=d.get("domain"),
            difficulty=Difficulty(d.get("difficulty", "easy")),
            relevance_grades=d.get("relevance_grades", {}),
            metadata=d.get("metadata", {}),
        )


@dataclass(frozen=True)
class GoldenSet:
    """A collection of golden queries for evaluation.

    Args:
        name: Human-readable name (e.g., "research-kb-v1").
        version: Semantic version string.
        queries: The golden queries.
        description: Optional description of this golden set.
    """

    name: str
    version: str
    queries: tuple[GoldenQuery, ...]
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("GoldenSet name must be non-empty")
        if not self.queries:
            raise ValueError("GoldenSet must contain at least one query")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        d: dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "queries": [q.to_dict() for q in self.queries],
        }
        if self.description:
            d["description"] = self.description
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GoldenSet:
        """Deserialize from dict."""
        return cls(
            name=d["name"],
            version=d["version"],
            queries=tuple(GoldenQuery.from_dict(q) for q in d["queries"]),
            description=d.get("description", ""),
        )

    def to_json(self, path: Path) -> None:
        """Write golden set to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> GoldenSet:
        """Load golden set from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def to_yaml(self, path: Path) -> None:
        """Write golden set to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: Path) -> GoldenSet:
        """Load golden set from YAML file."""
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    @classmethod
    def from_file(cls, path: Path) -> GoldenSet:
        """Load from JSON or YAML based on file extension."""
        suffix = path.suffix.lower()
        if suffix in (".yml", ".yaml"):
            return cls.from_yaml(path)
        return cls.from_json(path)

    def filter_by_domain(self, domain: str) -> GoldenSet:
        """Return a new GoldenSet with only queries matching the domain."""
        filtered = tuple(q for q in self.queries if q.domain == domain)
        if not filtered:
            raise ValueError(f"No queries match domain '{domain}'")
        return GoldenSet(
            name=f"{self.name}[{domain}]",
            version=self.version,
            queries=filtered,
            description=f"Filtered: {self.description}",
        )

    def filter_by_difficulty(self, difficulty: Difficulty) -> GoldenSet:
        """Return a new GoldenSet with only queries matching the difficulty."""
        filtered = tuple(q for q in self.queries if q.difficulty == difficulty)
        if not filtered:
            raise ValueError(f"No queries match difficulty '{difficulty.value}'")
        return GoldenSet(
            name=f"{self.name}[{difficulty.value}]",
            version=self.version,
            queries=filtered,
            description=f"Filtered: {self.description}",
        )


# ---------------------------------------------------------------------------
# Retrieval result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievedItem:
    """A single item returned by a retrieval system.

    Args:
        id: Unique identifier for the retrieved document/chunk.
        rank: 1-based rank position in results.
        score: Optional relevance score from the retrieval system.
        content: Optional text content (for display/debugging).
        metadata: Additional fields from the retrieval system.
    """

    id: str
    rank: int
    score: float | None = None
    content: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError(f"Rank must be >= 1, got {self.rank}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        d: dict[str, Any] = {"id": self.id, "rank": self.rank}
        if self.score is not None:
            d["score"] = self.score
        if self.content is not None:
            d["content"] = self.content
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RetrievedItem:
        """Deserialize from dict."""
        return cls(
            id=d["id"],
            rank=d["rank"],
            score=d.get("score"),
            content=d.get("content"),
            metadata=d.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Pre-computed result set (results-first evaluation path)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResultEntry:
    """Pre-computed retrieval results for a single query.

    Used with ``ResultSet`` to evaluate without a live retrieval adapter.

    Args:
        query: The query text (must match a GoldenQuery).
        retrieved: Ranked list of retrieved items.
    """

    query: str
    retrieved: tuple[RetrievedItem, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "query": self.query,
            "retrieved": [r.to_dict() for r in self.retrieved],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ResultEntry:
        """Deserialize from dict."""
        return cls(
            query=d["query"],
            retrieved=tuple(RetrievedItem.from_dict(r) for r in d["retrieved"]),
        )


@dataclass(frozen=True)
class ResultSet:
    """Pre-computed retrieval results for offline evaluation.

    The primary ``ir-eval`` evaluation path.  Pipe your retrieval system's
    output into a JSON file matching this schema, then run::

        ir-eval evaluate results.json --golden golden.json

    Args:
        name: Identifier for this result set (e.g. "my-rag-v2").
        results: One entry per query with ranked retrieval results.

    Example JSON::

        {
            "name": "my-rag-v2",
            "results": [
                {
                    "query": "instrumental variables",
                    "retrieved": [
                        {"id": "chunk_123", "rank": 1, "score": 0.95},
                        {"id": "chunk_456", "rank": 2, "score": 0.82}
                    ]
                }
            ]
        }
    """

    name: str
    results: tuple[ResultEntry, ...]

    def __post_init__(self) -> None:
        if not self.results:
            raise ValueError("ResultSet must contain at least one entry")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "name": self.name,
            "results": [r.to_dict() for r in self.results],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ResultSet:
        """Deserialize from dict."""
        return cls(
            name=d["name"],
            results=tuple(ResultEntry.from_dict(r) for r in d["results"]),
        )

    @classmethod
    def from_json(cls, path: Path) -> ResultSet:
        """Load from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def to_json(self, path: Path) -> None:
        """Write to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def lookup(self, query_text: str) -> list[RetrievedItem]:
        """Get retrieved items for a query, or empty list if not found."""
        for entry in self.results:
            if entry.query == query_text:
                return list(entry.retrieved)
        return []


# ---------------------------------------------------------------------------
# Evaluation result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryResult:
    """Per-query evaluation result with metrics.

    Args:
        query: The golden query that was evaluated.
        retrieved: Items returned by the retrieval system.
        hit: Whether any relevant document was found in top-k.
        reciprocal_rank: 1/rank of first relevant doc, or 0.0 if not found.
        ndcg: NDCG@k score for this query.
        precision_at_k: Precision@k (fraction of top-k that are relevant).
        average_precision: Average precision for this query.
        latency_ms: Optional retrieval latency in milliseconds.
    """

    query: GoldenQuery
    retrieved: tuple[RetrievedItem, ...]
    hit: bool = False
    reciprocal_rank: float = 0.0
    ndcg: float = 0.0
    precision_at_k: float = 0.0
    average_precision: float = 0.0
    latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        d: dict[str, Any] = {
            "query": self.query.to_dict(),
            "retrieved": [r.to_dict() for r in self.retrieved],
            "hit": self.hit,
            "reciprocal_rank": self.reciprocal_rank,
            "ndcg": self.ndcg,
            "precision_at_k": self.precision_at_k,
            "average_precision": self.average_precision,
        }
        if self.latency_ms is not None:
            d["latency_ms"] = self.latency_ms
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> QueryResult:
        """Deserialize from dict."""
        return cls(
            query=GoldenQuery.from_dict(d["query"]),
            retrieved=tuple(RetrievedItem.from_dict(r) for r in d["retrieved"]),
            hit=d["hit"],
            reciprocal_rank=d["reciprocal_rank"],
            ndcg=d["ndcg"],
            precision_at_k=d["precision_at_k"],
            average_precision=d["average_precision"],
            latency_ms=d.get("latency_ms"),
        )


@dataclass(frozen=True)
class EvalRun:
    """Complete evaluation run with aggregate metrics.

    Args:
        id: Unique run identifier (UUID or timestamp-based).
        golden_set_name: Name of the golden set used.
        adapter_name: Name of the retrieval adapter.
        timestamp: When the run was executed.
        query_results: Per-query results.
        metrics: Aggregate metrics (hit_rate, mrr, ndcg_at_5, etc.).
        config: Run configuration (top_k, adapter settings, etc.).
    """

    id: str
    golden_set_name: str
    adapter_name: str
    timestamp: datetime
    query_results: tuple[QueryResult, ...]
    metrics: dict[str, float]
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "id": self.id,
            "golden_set_name": self.golden_set_name,
            "adapter_name": self.adapter_name,
            "timestamp": self.timestamp.isoformat(),
            "query_results": [qr.to_dict() for qr in self.query_results],
            "metrics": self.metrics,
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvalRun:
        """Deserialize from dict."""
        return cls(
            id=d["id"],
            golden_set_name=d["golden_set_name"],
            adapter_name=d["adapter_name"],
            timestamp=datetime.fromisoformat(d["timestamp"]),
            query_results=tuple(QueryResult.from_dict(qr) for qr in d["query_results"]),
            metrics=d["metrics"],
            config=d.get("config", {}),
        )

    def to_json(self, path: Path) -> None:
        """Write run to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def from_json(cls, path: Path) -> EvalRun:
        """Load run from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Baseline and drift types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Baseline:
    """A pinned evaluation run used as comparison reference.

    Args:
        run: The evaluation run serving as baseline.
        set_at: When this baseline was set.
        set_by: Who/what set this baseline.
        notes: Optional notes about why this baseline was set.
    """

    run: EvalRun
    set_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    set_by: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "run": self.run.to_dict(),
            "set_at": self.set_at.isoformat(),
            "set_by": self.set_by,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Baseline:
        """Deserialize from dict."""
        return cls(
            run=EvalRun.from_dict(d["run"]),
            set_at=datetime.fromisoformat(d["set_at"]),
            set_by=d.get("set_by", ""),
            notes=d.get("notes", ""),
        )


@dataclass(frozen=True)
class DriftResult:
    """Per-metric comparison between baseline and current run.

    Args:
        metric_name: Name of the metric (e.g., "hit_rate", "mrr").
        baseline_value: Metric value from the baseline run.
        current_value: Metric value from the current run.
        delta: Absolute change (current - baseline).
        delta_pct: Percentage change ((current - baseline) / baseline * 100).
        p_value: Statistical significance of the change.
        ci_lower: Lower bound of confidence interval for the delta.
        ci_upper: Upper bound of confidence interval for the delta.
        severity: Classified severity of the drift.
        test_name: Name of the statistical test used.
    """

    metric_name: str
    baseline_value: float
    current_value: float
    delta: float
    delta_pct: float
    p_value: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    severity: DriftSeverity = DriftSeverity.INFO
    test_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        d: dict[str, Any] = {
            "metric_name": self.metric_name,
            "baseline_value": self.baseline_value,
            "current_value": self.current_value,
            "delta": self.delta,
            "delta_pct": self.delta_pct,
            "severity": self.severity.value,
        }
        if self.p_value is not None:
            d["p_value"] = self.p_value
        if self.ci_lower is not None:
            d["ci_lower"] = self.ci_lower
        if self.ci_upper is not None:
            d["ci_upper"] = self.ci_upper
        if self.test_name:
            d["test_name"] = self.test_name
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DriftResult:
        """Deserialize from dict."""
        return cls(
            metric_name=d["metric_name"],
            baseline_value=d["baseline_value"],
            current_value=d["current_value"],
            delta=d["delta"],
            delta_pct=d["delta_pct"],
            p_value=d.get("p_value"),
            ci_lower=d.get("ci_lower"),
            ci_upper=d.get("ci_upper"),
            severity=DriftSeverity(d.get("severity", "info")),
            test_name=d.get("test_name", ""),
        )
