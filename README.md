# ir-eval

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Statistical retrieval evaluation framework with golden-set metrics and drift detection.

> **ranx** is for papers. **ir-eval** is for CI/CD pipelines.

Most RAG systems fail silently. ir-eval catches regression before users do — with paired statistical tests, not LLM-as-judge.

## Why ir-eval?

| Feature | ir-eval | RAGAS | DeepEval | ranx |
|---------|---------|-------|----------|------|
| Deterministic (no LLM-as-judge) | Yes | No | No | Yes |
| Results-first (no infra needed) | Yes | No | No | Yes |
| Paired statistical tests | Yes | No | No | No |
| Bootstrap confidence intervals | Yes | No | No | No |
| Drift detection with severity | Yes | No | Partial | No |
| CI/CD exit codes | Yes | No | Partial | No |
| Zero torch/sklearn dependency | Yes | No | No | No |

## Quick Start — Results-First (Primary Path)

Export your retrieval system's results to JSON, then evaluate:

```bash
# 1. Validate your golden set
ir-eval validate golden.json

# 2. Evaluate pre-computed results
ir-eval evaluate results.json --golden golden.json

# 3. Set baseline
ir-eval baseline set run_output.json --notes "v1.0"

# 4. Detect drift in CI
ir-eval evaluate new_results.json --golden golden.json --output new_run.json
ir-eval drift golden.json --adapter my-adapter --exit-code
```

**results.json** format:

```json
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
```

## Quick Start — Adapter (Live Evaluation)

For live evaluation against a running retrieval system:

```python
from ir_eval import GoldenSet, RetrievedItem
from ir_eval.runner import run_evaluation

# 1. Load your golden set
golden = GoldenSet.from_json("golden.json")

# 2. Implement the adapter protocol (2 methods)
class MyAdapter:
    @property
    def name(self) -> str:
        return "my-rag-system"

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedItem]:
        results = my_search(query, limit=top_k)
        return [RetrievedItem(id=r.id, rank=i+1, score=r.score)
                for i, r in enumerate(results)]

# 3. Run evaluation
run = run_evaluation(golden, MyAdapter(), top_k=10)
print(f"Hit Rate: {run.metrics['hit_rate']:.1%}")
print(f"MRR:      {run.metrics['mrr']:.3f}")
print(f"NDCG@k:   {run.metrics['ndcg_at_k']:.3f}")
```

## Installation

```bash
pip install ir-eval

# With optional scipy (faster Fisher exact test)
pip install "ir-eval[scipy]"
```

## Metrics

### Ranking Metrics
- **MRR** (Mean Reciprocal Rank): Average of 1/rank of first relevant document
- **NDCG@k**: Normalized Discounted Cumulative Gain (supports graded relevance)
- **Hit Rate@k**: Fraction of queries with at least one relevant result in top-k
- **Precision@k**: Fraction of top-k results that are relevant
- **MAP** (Mean Average Precision): Area under precision-recall curve

### Statistical Tests
- **Bootstrap CI**: Non-parametric confidence intervals for any metric
- **Paired permutation test**: Compare two systems on the same queries
- **Fisher's exact test**: 2x2 contingency for small samples (n < 50)
- **McNemar's test**: Paired binary outcomes (hit/miss changes)

### Drift Detection
- Compares current run against a stored baseline
- 2D severity classification (magnitude x statistical significance):
  - **INFO**: No significant change
  - **WARNING**: >5% drop AND p < 0.10
  - **CRITICAL**: >10% drop AND p < 0.05

## For Statisticians

The key design choice: **per-query metric storage enables paired tests**.

Most evaluation frameworks store only aggregate metrics, forcing unpaired tests
(chi-squared, two-sample z-test) that are far less powerful. By storing per-query
results, ir-eval enables:

1. **Paired permutation test** for continuous metrics (MRR, NDCG) — no
   distributional assumptions, exact p-values
2. **McNemar's test** for hit rate — proper test for paired binary outcomes
3. **Bootstrap CI** — works for any metric without normality assumption
4. **Fisher's exact test** — chosen over chi-squared because golden sets
   are typically 20-100 queries, where chi-squared expected cell counts < 5

## Architecture

```
Results JSON ─→ Runner ─→ EvalRun ─→ Reporter
Golden Set ──┘              │
                            ├─→ Baseline Store
                            └─→ Drift Detector ─→ Alerts (exit codes)
```

Two evaluation paths:
- **Results-first** (primary): Load pre-computed results from JSON — no infrastructure needed
- **Adapter** (live): Implement a 2-method Protocol for live retrieval

See [docs/design.md](docs/design.md) for details.

## CI Integration

```yaml
# .github/workflows/eval.yml
- name: Evaluate retrieval quality
  run: ir-eval evaluate results.json --golden golden.json --output run.json

- name: Check for drift
  run: ir-eval drift golden.json --adapter my-adapter --exit-code
```

The `--exit-code` flag exits 1 on CRITICAL drift, failing your CI pipeline
when retrieval quality regresses significantly.

## CLI Reference

```bash
ir-eval evaluate results.json --golden golden.json   # Primary: evaluate pre-computed results
ir-eval run golden.json --adapter name               # Live: evaluate via adapter
ir-eval validate golden.json                          # Validate golden set structure
ir-eval compare run_a.json run_b.json                 # Compare two runs side-by-side
ir-eval drift golden.json --adapter name --exit-code  # Detect regression from baseline
ir-eval baseline set run.json --notes "v1.0"          # Set baseline for drift detection
ir-eval baseline show golden-set-name                 # Show current baseline
ir-eval history golden-set-name                       # Show baseline history
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

MIT
