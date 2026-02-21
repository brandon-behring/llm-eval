"""Tests for statistical confidence: bootstrap CI, Fisher exact, McNemar.

Uses known-value tests and statistical property tests.
"""

from __future__ import annotations

import pytest

from ir_eval.metrics.confidence import (
    bootstrap_ci,
    fisher_exact_test,
    mcnemar_test,
    paired_bootstrap_test,
)


class TestBootstrapCI:
    """Bootstrap confidence intervals."""

    def test_ci_covers_known_mean(self) -> None:
        """95% CI should cover the true mean most of the time."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        ci = bootstrap_ci(values, seed=42)
        assert ci.lower <= 5.5 <= ci.upper

    def test_ci_width_decreases_with_n(self) -> None:
        """Larger sample from same distribution → narrower relative CI.

        Use values from same range to isolate the sample size effect.
        """
        import random as rng

        rng.seed(99)
        small = [rng.uniform(0, 1) for _ in range(10)]
        large = [rng.uniform(0, 1) for _ in range(200)]
        ci_small = bootstrap_ci(small, seed=42)
        ci_large = bootstrap_ci(large, seed=42)
        # Relative width (width / estimate) should be smaller for large n
        width_small = ci_small.upper - ci_small.lower
        width_large = ci_large.upper - ci_large.lower
        assert width_large < width_small

    def test_deterministic_with_seed(self) -> None:
        """Same seed → same CI."""
        values = [1.0, 2.0, 3.0]
        ci1 = bootstrap_ci(values, seed=123)
        ci2 = bootstrap_ci(values, seed=123)
        assert ci1.lower == ci2.lower
        assert ci1.upper == ci2.upper

    def test_custom_statistic(self) -> None:
        """Custom statistic function (median)."""
        values = [1.0, 2.0, 100.0]  # Skewed
        def median_fn(v):
            return sorted(v)[len(v) // 2]
        ci = bootstrap_ci(values, statistic_fn=median_fn, seed=42)
        # Median of [1, 2, 100] = 2
        assert ci.estimate == 2.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            bootstrap_ci([])

    def test_confidence_level_stored(self) -> None:
        ci = bootstrap_ci([1.0, 2.0], confidence_level=0.90, seed=42)
        assert ci.confidence_level == 0.90

    def test_single_value(self) -> None:
        """Single value → CI collapses to that value."""
        ci = bootstrap_ci([5.0], seed=42)
        assert ci.estimate == 5.0
        assert ci.lower == 5.0
        assert ci.upper == 5.0


class TestPairedBootstrapTest:
    """Paired bootstrap hypothesis test."""

    def test_identical_systems_not_significant(self) -> None:
        """Two identical systems → p ≈ 1.0 (not significant)."""
        values = [0.5, 0.7, 0.3, 0.9, 0.6]
        result = paired_bootstrap_test(values, values, seed=42)
        assert not result.significant
        assert result.p_value > 0.5

    def test_clearly_different_systems(self) -> None:
        """Dramatically different with variance → should be significant.

        Note: constant values make every resample produce the same diff,
        so we need some variance for the bootstrap to detect significance.
        """
        a = [0.1, 0.15, 0.05, 0.12, 0.08, 0.11, 0.09, 0.13, 0.07, 0.10]
        b = [0.9, 0.85, 0.95, 0.88, 0.92, 0.89, 0.91, 0.87, 0.93, 0.90]
        result = paired_bootstrap_test(a, b, seed=42)
        assert result.significant
        assert result.p_value < 0.01

    def test_effect_size_positive(self) -> None:
        """B better than A → positive effect size."""
        a = [0.3, 0.3, 0.3]
        b = [0.7, 0.7, 0.7]
        result = paired_bootstrap_test(a, b, seed=42)
        assert result.effect_size is not None
        assert result.effect_size > 0

    def test_different_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            paired_bootstrap_test([1.0], [1.0, 2.0])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            paired_bootstrap_test([], [])


class TestFisherExactTest:
    """Fisher's exact test for 2x2 tables."""

    def test_known_table(self) -> None:
        """Classic Lady Tasting Tea example.

        Table: [[1, 9], [11, 3]]
        Expected p-value is small.
        """
        result = fisher_exact_test(1, 11, 9, 3)
        assert result.test_name == "fisher_exact"
        assert result.p_value < 0.05

    def test_equal_proportions_not_significant(self) -> None:
        """Equal success rates → not significant."""
        result = fisher_exact_test(10, 10, 10, 10)
        assert not result.significant
        assert result.p_value > 0.5

    def test_extreme_difference(self) -> None:
        """All pass vs all fail → highly significant."""
        result = fisher_exact_test(20, 0, 0, 20)
        assert result.significant
        assert result.p_value < 0.001


class TestMcNemarTest:
    """McNemar's test for paired binary outcomes."""

    def test_no_discordant_pairs(self) -> None:
        """No disagreements → p = 1.0."""
        result = mcnemar_test(both_correct=10, a_only=0, b_only=0, both_wrong=5)
        assert result.p_value == 1.0
        assert not result.significant

    def test_symmetric_discordant(self) -> None:
        """Equal discordant pairs → not significant."""
        result = mcnemar_test(both_correct=10, a_only=5, b_only=5, both_wrong=5)
        assert not result.significant
        assert result.p_value == 1.0  # Symmetric binomial

    def test_asymmetric_discordant_significant(self) -> None:
        """Strongly asymmetric → should be significant.

        a_only=1, b_only=10 → B clearly better.
        P(X<=1) under Binom(11, 0.5) ≈ 0.006
        Two-sided p ≈ 0.012
        """
        result = mcnemar_test(both_correct=20, a_only=1, b_only=10, both_wrong=5)
        assert result.significant
        assert result.p_value < 0.05

    def test_effect_size(self) -> None:
        """Effect size should reflect direction of improvement."""
        result = mcnemar_test(both_correct=10, a_only=2, b_only=8, both_wrong=5)
        assert result.effect_size is not None
        assert result.effect_size > 0  # B better than A
