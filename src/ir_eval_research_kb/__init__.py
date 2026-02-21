"""Optional research-kb adapter for ir-eval.

Provides ResearchKBAdapter that wraps the research-kb hybrid search.

Usage:
    from ir_eval_research_kb import ResearchKBAdapter

    adapter = ResearchKBAdapter()
    run = run_evaluation(golden_set, adapter)
"""

from ir_eval_research_kb.adapter import ResearchKBAdapter

__all__ = ["ResearchKBAdapter"]
