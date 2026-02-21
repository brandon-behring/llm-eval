"""Tests for CLI commands."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from ir_eval.cli.main import app

runner = CliRunner()


class TestValidateCommand:
    """ir-eval validate <golden-set>."""

    def test_validate_valid_file(self, golden_small_path: Path) -> None:
        result = runner.invoke(app, ["validate", str(golden_small_path)])
        assert result.exit_code == 0
        assert "Valid golden set" in result.output
        assert "10" in result.output  # 10 queries

    def test_validate_research_kb(self, golden_research_kb_path: Path) -> None:
        result = runner.invoke(app, ["validate", str(golden_research_kb_path)])
        assert result.exit_code == 0
        assert "47" in result.output

    def test_validate_invalid_file(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text('{"invalid": true}')
        result = runner.invoke(app, ["validate", str(bad)])
        assert result.exit_code == 1


class TestHistoryCommand:
    """ir-eval history <golden-set-name>."""

    def test_no_baselines(self, tmp_path: Path) -> None:
        result = runner.invoke(app, [
            "history", "nonexistent",
            "--store-dir", str(tmp_path / "baselines"),
        ])
        assert "No baselines found" in result.output


class TestBaselineCommands:
    """ir-eval baseline set/show."""

    def test_set_and_show(self, sample_eval_run, tmp_path: Path) -> None:

        run_path = tmp_path / "run.json"
        sample_eval_run.to_json(run_path)
        store_dir = tmp_path / "baselines"

        # Set
        result = runner.invoke(app, [
            "baseline", "set", str(run_path),
            "--notes", "test baseline",
            "--store-dir", str(store_dir),
        ])
        assert result.exit_code == 0
        assert "Baseline set" in result.output

        # Show
        result = runner.invoke(app, [
            "baseline", "show", "test-set",
            "--store-dir", str(store_dir),
        ])
        assert result.exit_code == 0
        assert "test-run-001" in result.output

    def test_show_nonexistent(self, tmp_path: Path) -> None:
        result = runner.invoke(app, [
            "baseline", "show", "nonexistent",
            "--store-dir", str(tmp_path / "baselines"),
        ])
        assert result.exit_code == 1
        assert "No baseline" in result.output


class TestCompareCommand:
    """ir-eval compare <run-a> <run-b>."""

    def test_compare_same_run(self, sample_eval_run, tmp_path: Path) -> None:
        path_a = tmp_path / "a.json"
        path_b = tmp_path / "b.json"
        sample_eval_run.to_json(path_a)
        sample_eval_run.to_json(path_b)

        result = runner.invoke(app, ["compare", str(path_a), str(path_b)])
        assert result.exit_code == 0

    def test_compare_json_format(self, sample_eval_run, tmp_path: Path) -> None:
        path_a = tmp_path / "a.json"
        path_b = tmp_path / "b.json"
        sample_eval_run.to_json(path_a)
        sample_eval_run.to_json(path_b)

        result = runner.invoke(app, [
            "compare", str(path_a), str(path_b), "--format", "json",
        ])
        assert result.exit_code == 0
        import json

        data = json.loads(result.output)
        assert "metric_deltas" in data

    def test_compare_markdown_format(self, sample_eval_run, tmp_path: Path) -> None:
        path_a = tmp_path / "a.json"
        path_b = tmp_path / "b.json"
        sample_eval_run.to_json(path_a)
        sample_eval_run.to_json(path_b)

        result = runner.invoke(app, [
            "compare", str(path_a), str(path_b), "--format", "markdown",
        ])
        assert result.exit_code == 0
        assert "# Comparison" in result.output
