"""File-based baseline storage.

Baselines are stored as JSON files in `.ir-eval/baselines/`.
Each golden set has at most one active baseline.

Usage:
    store = BaselineStore()
    store.set_baseline(eval_run, notes="initial baseline")
    baseline = store.get_baseline("research-kb-v1")
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ir_eval.types import Baseline, EvalRun


class BaselineStore:
    """File-based storage for evaluation baselines.

    Args:
        base_dir: Directory for baseline files. Defaults to `.ir-eval/baselines/`.
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = base_dir or Path(".ir-eval") / "baselines"

    def _ensure_dir(self) -> None:
        """Create the baselines directory if needed."""
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, golden_set_name: str) -> Path:
        """Get the file path for a golden set's baseline."""
        safe_name = golden_set_name.replace("/", "_").replace(" ", "_")
        return self._base_dir / f"{safe_name}.json"

    def set_baseline(
        self,
        run: EvalRun,
        set_by: str = "",
        notes: str = "",
    ) -> Baseline:
        """Set or update the baseline for a golden set.

        Args:
            run: The evaluation run to use as baseline.
            set_by: Who/what set this baseline.
            notes: Optional notes.

        Returns:
            The created Baseline object.
        """
        self._ensure_dir()

        baseline = Baseline(
            run=run,
            set_at=datetime.now(UTC),
            set_by=set_by,
            notes=notes,
        )

        path = self._path_for(run.golden_set_name)
        with open(path, "w") as f:
            json.dump(baseline.to_dict(), f, indent=2, default=str)

        return baseline

    def get_baseline(self, golden_set_name: str) -> Baseline | None:
        """Get the current baseline for a golden set.

        Args:
            golden_set_name: Name of the golden set.

        Returns:
            Baseline if one exists, None otherwise.
        """
        path = self._path_for(golden_set_name)
        if not path.exists():
            return None

        with open(path) as f:
            data = json.load(f)

        return Baseline.from_dict(data)

    def list_baselines(self) -> list[dict[str, Any]]:
        """List all stored baselines.

        Returns:
            List of dicts with golden_set_name, adapter, timestamp, notes.
        """
        if not self._base_dir.exists():
            return []

        baselines = []
        for path in sorted(self._base_dir.glob("*.json")):
            with open(path) as f:
                data = json.load(f)
            baselines.append({
                "golden_set_name": data["run"]["golden_set_name"],
                "adapter_name": data["run"]["adapter_name"],
                "timestamp": data["run"]["timestamp"],
                "set_at": data["set_at"],
                "notes": data.get("notes", ""),
                "file": str(path),
            })

        return baselines

    def delete_baseline(self, golden_set_name: str) -> bool:
        """Delete a baseline.

        Args:
            golden_set_name: Name of the golden set.

        Returns:
            True if deleted, False if not found.
        """
        path = self._path_for(golden_set_name)
        if path.exists():
            path.unlink()
            return True
        return False
