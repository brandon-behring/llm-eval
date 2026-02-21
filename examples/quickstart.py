#!/usr/bin/env python3
"""Quickstart example: evaluate a stub adapter in 20 lines.

Run:
    python examples/quickstart.py
"""

from ir_eval import GoldenQuery, GoldenSet, RetrievedItem
from ir_eval.runner import run_evaluation


class MyAdapter:
    """Minimal adapter that always returns doc-a at rank 1."""

    @property
    def name(self) -> str:
        return "my-adapter"

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedItem]:
        return [RetrievedItem(id="doc-a", rank=1, score=0.95)]


# Build a tiny golden set
golden = GoldenSet(
    name="demo",
    version="1.0",
    queries=(
        GoldenQuery(query="what is X?", relevant_ids=("doc-a",)),
        GoldenQuery(query="what is Y?", relevant_ids=("doc-b",)),
        GoldenQuery(query="what is Z?", relevant_ids=("doc-a", "doc-c")),
    ),
)

# Run evaluation
run = run_evaluation(golden, MyAdapter(), top_k=5)

# Print results
print(f"Hit Rate: {run.metrics['hit_rate']:.1%}")
print(f"MRR:      {run.metrics['mrr']:.3f}")
print(f"NDCG@5:   {run.metrics['ndcg_at_k']:.3f}")
print(f"MAP:      {run.metrics['map']:.3f}")
