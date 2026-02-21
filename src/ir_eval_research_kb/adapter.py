"""Research-kb retrieval adapter for ir-eval.

Wraps the research-kb hybrid search (FTS + vector) as a synchronous
RetrievalAdapter. Maintains a persistent event loop internally since
research-kb is async.

Requirements:
    - research-kb packages on PYTHONPATH (storage, pdf-tools, common, contracts)
    - PostgreSQL with research-kb data
    - BGE-large embedding model accessible

Usage:
    from ir_eval_research_kb import ResearchKBAdapter
    from ir_eval.runner import run_evaluation

    adapter = ResearchKBAdapter()
    run = run_evaluation(golden_set, adapter, top_k=10)
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from ir_eval.types import RetrievedItem


class ResearchKBAdapter:
    """Adapter bridging research-kb search to ir-eval.

    Maintains a persistent asyncio event loop so that the connection pool
    created during initialization remains valid for subsequent queries.

    Args:
        fts_weight: Full-text search weight (default 0.3).
        vector_weight: Vector similarity weight (default 0.7).
        research_kb_root: Path to research-kb repo root.
    """

    def __init__(
        self,
        fts_weight: float = 0.3,
        vector_weight: float = 0.7,
        research_kb_root: Path | None = None,
    ) -> None:
        self._fts_weight = fts_weight
        self._vector_weight = vector_weight
        self._initialized = False
        self._loop: asyncio.AbstractEventLoop | None = None

        # Add research-kb packages to path
        root = research_kb_root or Path.home() / "Claude" / "research-kb"
        for pkg in ["pdf-tools", "storage", "common", "contracts"]:
            pkg_path = str(root / "packages" / pkg / "src")
            if pkg_path not in sys.path:
                sys.path.insert(0, pkg_path)

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create the persistent event loop."""
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def _ensure_init(self) -> None:
        """Lazy-initialize async resources on a persistent event loop."""
        if not self._initialized:
            from research_kb_pdf import EmbeddingClient  # type: ignore[import-not-found]
            from research_kb_storage import (  # type: ignore[import-not-found]
                DatabaseConfig,
                get_connection_pool,
            )

            self._embed_client = EmbeddingClient()
            self._db_config = DatabaseConfig()

            loop = self._get_loop()
            loop.run_until_complete(get_connection_pool(self._db_config))
            self._initialized = True

    @property
    def name(self) -> str:
        """Adapter name for reports."""
        return "research-kb"

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedItem]:
        """Execute hybrid search against research-kb.

        Args:
            query: Search query text.
            top_k: Number of results.

        Returns:
            Ranked list of RetrievedItem.
        """
        self._ensure_init()

        from research_kb_storage import SearchQuery, search_hybrid  # type: ignore[import-not-found]

        query_embedding = self._embed_client.embed_query(query)

        search_query = SearchQuery(
            text=query,
            embedding=query_embedding,
            fts_weight=self._fts_weight,
            vector_weight=self._vector_weight,
            limit=top_k,
        )

        loop = self._get_loop()
        results = loop.run_until_complete(search_hybrid(search_query))

        return [
            RetrievedItem(
                id=str(r.chunk.id),
                rank=r.rank,
                score=getattr(r, "score", None),
                content=getattr(r.chunk, "content", None),
                metadata={
                    "source_title": r.source.title,
                    "page_start": getattr(r.chunk, "page_start", None),
                },
            )
            for r in results
        ]

    def close(self) -> None:
        """Close the event loop and release resources."""
        if self._loop is not None and not self._loop.is_closed():
            self._loop.close()
            self._loop = None
        self._initialized = False

    def __del__(self) -> None:
        self.close()
