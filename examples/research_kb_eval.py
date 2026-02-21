#!/usr/bin/env python3
"""Full research-kb evaluation with drift detection.

Demonstrates:
    1. Loading the 177-query golden set (14 domains)
    2. Running evaluation against research-kb
    3. Setting a baseline
    4. Running drift detection with statistical tests

Requirements:
    - research-kb running (PostgreSQL + embeddings)
    - Golden set at tests/fixtures/golden_research_kb_full.json

Run:
    python examples/research_kb_eval.py
"""

from pathlib import Path

from ir_eval import GoldenSet
from ir_eval.drift.baseline import BaselineStore
from ir_eval.drift.detector import DriftDetector
from ir_eval.reporters import console as console_reporter
from ir_eval.runner import run_evaluation

# Optional: only import if research-kb is available
try:
    from ir_eval_research_kb import ResearchKBAdapter

    HAS_RESEARCH_KB = True
except ImportError:
    HAS_RESEARCH_KB = False


def main() -> None:
    """Run the research-kb evaluation demo."""
    fixtures_dir = Path(__file__).parent.parent / "tests" / "fixtures"
    golden_path = fixtures_dir / "golden_research_kb_full.json"

    if not golden_path.exists():
        print(f"Golden set not found: {golden_path}")
        return

    golden = GoldenSet.from_json(golden_path)
    print(f"Loaded golden set: {golden.name} ({len(golden.queries)} queries)")

    if not HAS_RESEARCH_KB:
        print("\nresearch-kb adapter not available.")
        print("Install with: pip install -e '.[research-kb]'")
        print("\nTo run without research-kb, use the quickstart example instead.")
        return

    # Run evaluation
    adapter = ResearchKBAdapter()
    run = run_evaluation(golden, adapter, top_k=10)
    console_reporter.report_eval_run(run)

    # Expected baseline metrics (from prior evaluation)
    print("\nExpected metrics (known baseline):")
    print("  Hit Rate: ~92.9%")
    print("  MRR:      ~0.849")
    print("  NDCG@5:   ~0.823")

    # Save as baseline
    store = BaselineStore()
    baseline = store.set_baseline(run, set_by="demo", notes="research-kb-eval demo run")
    print(f"\nBaseline saved for '{golden.name}'")

    # Run drift detection (against itself — should be all INFO)
    detector = DriftDetector(seed=42)
    drift_results = detector.detect(baseline.run, run)
    console_reporter.report_drift(drift_results)


if __name__ == "__main__":
    main()
