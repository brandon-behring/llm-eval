"""Retrieval adapter protocol for ir-eval.

Any retrieval system (LangChain, LlamaIndex, Elasticsearch, custom) implements
this protocol to be evaluated by ir-eval. The interface is deliberately
synchronous — async systems wrap with asyncio.run() internally.

Adapters register via entry points:
    [project.entry-points."ir_eval.adapters"]
    my_adapter = "my_package:MyAdapter"
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ir_eval.types import RetrievedItem


@runtime_checkable
class RetrievalAdapter(Protocol):
    """Protocol for retrieval systems to implement.

    Example:
        class MyAdapter:
            @property
            def name(self) -> str:
                return "my-retrieval-system"

            def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedItem]:
                results = my_search(query, limit=top_k)
                return [
                    RetrievedItem(id=r.id, rank=i+1, score=r.score)
                    for i, r in enumerate(results)
                ]
    """

    @property
    def name(self) -> str:
        """Human-readable name for this adapter (used in reports)."""
        ...

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedItem]:
        """Execute a retrieval query and return ranked results.

        Args:
            query: The search query text.
            top_k: Maximum number of results to return.

        Returns:
            List of RetrievedItem, ordered by rank (1-based).
        """
        ...
