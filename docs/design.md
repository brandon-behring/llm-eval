# Architecture & Design Decisions

## Core Architecture

```
Golden Set ─→ Adapter Protocol ─→ Runner ─→ EvalRun
                                      │
                                      ├─→ Baseline Store (.ir-eval/baselines/)
                                      ├─→ Drift Detector (statistical tests)
                                      ├─→ Reporters (console/markdown/JSON)
                                      └─→ CLI (Typer)
```

## Key Design Decisions

### Golden-set-based, not LLM-as-judge

**Decision**: Evaluate against human-curated golden sets with known relevant documents.

**Rationale**: Deterministic, reproducible, cheap to run. LLM-as-judge (RAGAS-style)
introduces non-determinism, is expensive at scale, and conflates evaluator quality
with system quality.

**Tradeoff**: Requires upfront effort to build golden sets. Worth it for systems
where regression detection matters.

### Per-query metric storage

**Decision**: Store all metrics at the query level, not just aggregates.

**Rationale**: Enables paired statistical tests. Comparing two systems on the
same 47 queries with a paired test is far more powerful than comparing two
aggregate numbers.

### Fisher exact over chi-squared

**Decision**: Use Fisher's exact test for hit rate comparison.

**Rationale**: Golden sets are typically 20-100 queries. At n=47, chi-squared
expected cell counts can fall below 5, making the approximation unreliable.
Fisher's exact test is... exact.

### Permutation test over t-test

**Decision**: Use paired permutation test for continuous metrics (MRR, NDCG).

**Rationale**: MRR values are bounded [0, 1] and highly skewed (many 1.0 and 0.0
values). The Central Limit Theorem doesn't apply well to small skewed samples.
Permutation tests make no distributional assumptions.

### Bootstrap CI for everything

**Decision**: Use bootstrap percentile CIs rather than normal approximation.

**Rationale**: Works for MRR, NDCG, Hit Rate without assuming normality.
These metrics are bounded, often skewed, and sometimes multimodal.

### Sync adapter protocol

**Decision**: `RetrievalAdapter.retrieve()` is synchronous.

**Rationale**: Most RAG systems (LangChain, LlamaIndex, Elasticsearch) have
sync APIs. Async systems wrap with `asyncio.run()` internally (as our
research-kb adapter does). Forcing async on sync users is worse than the reverse.

### No torch/sklearn dependency

**Decision**: Pure Python implementation of all metrics.

**Rationale**: NDCG is 5 lines of code. MRR is 3 lines. Pulling in sklearn
(which pulls numpy, scipy) for these trivial computations is unnecessary.
The only optional dependency is scipy for a faster Fisher exact test.

### Hatchling build backend

**Decision**: Use hatchling with PEP 621 metadata.

**Rationale**: Simpler than Poetry for a standalone package. No lock file
complexity. `pyproject.toml` is the single source of truth.

## Severity Classification

Drift severity is a 2D classification based on:
1. **Magnitude**: How much did the metric change?
2. **Significance**: Is the change statistically significant?

| Magnitude | p < 0.05 | p < 0.10 | p >= 0.10 |
|-----------|----------|----------|-----------|
| > 10% drop | CRITICAL | CRITICAL | WARNING |
| > 5% drop | WARNING | WARNING | INFO |
| <= 5% drop | INFO | INFO | INFO |
| Any increase | INFO | INFO | INFO |

This avoids two failure modes:
- Flagging tiny but statistically significant changes (noise)
- Flagging large changes from random variation (insufficient evidence)

## Extension Points

### Custom Adapters

Register via entry points in `pyproject.toml`:

```toml
[project.entry-points."ir_eval.adapters"]
my-adapter = "my_package:MyAdapter"
```

### Custom Reporters

Implement the same signature as `reporters/console.py`.

### Custom Alert Sinks

Implement the `AlertSink` protocol for Slack, email, PagerDuty, etc.
