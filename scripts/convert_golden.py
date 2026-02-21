#!/usr/bin/env python3
"""Convert research-kb golden dataset to ir-eval GoldenSet format.

Transforms:
    - Flat list → GoldenSet envelope (name, version, description, queries)
    - target_chunk_ids → relevant_ids (per query)
    - source_title → metadata.source_title (per query)

Usage:
    python scripts/convert_golden.py <input_path> [output_path]
    python scripts/convert_golden.py ~/Claude/research-kb/fixtures/eval/golden_dataset.json

If output_path is omitted, writes to tests/fixtures/golden_research_kb_full.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def convert_golden(input_path: Path, output_path: Path) -> int:
    """Convert research-kb golden dataset to ir-eval GoldenSet format.

    Parameters
    ----------
    input_path : Path
        Path to research-kb golden_dataset.json (flat list of queries).
    output_path : Path
        Path for ir-eval GoldenSet JSON output.

    Returns
    -------
    int
        Number of queries converted.

    Raises
    ------
    FileNotFoundError
        If input_path does not exist.
    ValueError
        If input is not a list or queries lack required fields.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path) as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError(f"Expected list, got {type(raw).__name__}")

    queries = []
    domains: set[str] = set()

    for i, entry in enumerate(raw):
        if "query" not in entry:
            raise ValueError(f"Query {i} missing 'query' field")
        if "target_chunk_ids" not in entry:
            raise ValueError(f"Query {i} missing 'target_chunk_ids' field")

        domain = entry.get("domain", "unknown")
        domains.add(domain)

        query = {
            "query": entry["query"],
            "relevant_ids": entry["target_chunk_ids"],
            "domain": domain,
            "difficulty": entry.get("difficulty", "medium"),
            "metadata": {},
        }

        if "source_title" in entry:
            query["metadata"]["source_title"] = entry["source_title"]

        queries.append(query)

    golden_set = {
        "name": "research-kb-v1",
        "version": "2.0.0",
        "description": (
            f"{len(queries)}-query golden set converted from research-kb evaluation fixtures. "
            f"Covers {len(domains)} domains: {', '.join(sorted(domains))}."
        ),
        "queries": queries,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(golden_set, f, indent=2)

    return len(queries)


def main() -> None:
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/convert_golden.py <input_path> [output_path]")
        sys.exit(1)

    input_path = Path(sys.argv[1]).expanduser()
    default_output = Path(__file__).parent.parent / "tests" / "fixtures" / "golden_research_kb_full.json"
    output_path = Path(sys.argv[2]).expanduser() if len(sys.argv) > 2 else default_output

    count = convert_golden(input_path, output_path)
    print(f"Converted {count} queries → {output_path}")


if __name__ == "__main__":
    main()
