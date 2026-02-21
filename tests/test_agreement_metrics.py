"""Tests for agreement metrics: Cohen's Kappa, weighted Kappa, confusion matrix."""

from __future__ import annotations

import pytest

from ir_eval.metrics.agreement import (
    agreement_rate,
    cohens_kappa,
    confusion_matrix,
    format_confusion_matrix,
    weighted_kappa,
)


class TestAgreementRate:
    """Simple percentage agreement."""

    def test_perfect_agreement(self) -> None:
        assert agreement_rate(["a", "b", "c"], ["a", "b", "c"]) == 1.0

    def test_no_agreement(self) -> None:
        assert agreement_rate(["a", "b", "c"], ["c", "a", "b"]) == 0.0

    def test_partial_agreement(self) -> None:
        assert agreement_rate(["a", "b", "c", "d"], ["a", "b", "x", "y"]) == 0.5

    def test_different_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            agreement_rate(["a"], ["a", "b"])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            agreement_rate([], [])


class TestCohensKappa:
    """Cohen's Kappa for nominal categories."""

    def test_perfect_agreement(self) -> None:
        """All items agree → kappa = 1.0."""
        r1 = ["a", "b", "a", "b"]
        r2 = ["a", "b", "a", "b"]
        assert cohens_kappa(r1, r2) == pytest.approx(1.0)

    def test_chance_agreement(self) -> None:
        """When agreement is at chance level → kappa ≈ 0."""
        # Two raters independently assign 50/50 a/b
        # They agree by chance 50% of the time
        r1 = ["a", "a", "b", "b"]
        r2 = ["a", "b", "a", "b"]
        # p_o = 2/4 = 0.5
        # p_e = (2*2 + 2*2) / 16 = 0.5
        # kappa = (0.5 - 0.5) / (1 - 0.5) = 0.0
        assert cohens_kappa(r1, r2) == pytest.approx(0.0)

    def test_known_kappa_value(self) -> None:
        """Hand-calculated known value.

        r1: [a, a, a, b, b, b, c, c, c, c]
        r2: [a, a, b, b, b, c, c, c, c, a]
        Matches: positions 0,1,3,4,6,7,8 → 7/10

        p_o = 7/10 = 0.7
        counts1: a=3, b=3, c=4
        counts2: a=3, b=3, c=4
        p_e = (3*3 + 3*3 + 4*4) / 100 = (9+9+16)/100 = 0.34
        kappa = (0.7 - 0.34) / (1 - 0.34) = 0.5454
        """
        r1 = ["a", "a", "a", "b", "b", "b", "c", "c", "c", "c"]
        r2 = ["a", "a", "b", "b", "b", "c", "c", "c", "c", "a"]
        kappa = cohens_kappa(r1, r2)
        assert kappa == pytest.approx(0.36 / 0.66, abs=0.001)

    def test_all_same_category(self) -> None:
        """Both raters assign same category to everything → kappa = 1.0."""
        r1 = ["a", "a", "a"]
        r2 = ["a", "a", "a"]
        assert cohens_kappa(r1, r2) == 1.0

    def test_different_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            cohens_kappa(["a"], ["a", "b"])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            cohens_kappa([], [])


class TestWeightedKappa:
    """Weighted Kappa for ordinal categories."""

    def test_perfect_agreement(self) -> None:
        cats = ["low", "mid", "high"]
        r1 = ["low", "mid", "high"]
        r2 = ["low", "mid", "high"]
        assert weighted_kappa(r1, r2, cats) == pytest.approx(1.0)

    def test_adjacent_disagreement_penalized_less(self) -> None:
        """Adjacent disagreement should give higher kappa than distant."""
        cats = ["low", "mid", "high"]
        # Adjacent disagreement
        r1_adj = ["low", "mid", "high"]
        r2_adj = ["mid", "high", "high"]  # off by 1
        kappa_adj = weighted_kappa(r1_adj, r2_adj, cats, weight="linear")

        # Distant disagreement
        r1_far = ["low", "mid", "high"]
        r2_far = ["high", "low", "high"]  # off by 2
        kappa_far = weighted_kappa(r1_far, r2_far, cats, weight="linear")

        assert kappa_adj > kappa_far

    def test_quadratic_weight(self) -> None:
        cats = ["a", "b", "c", "d"]
        r1 = ["a", "b", "c", "d"]
        r2 = ["a", "b", "c", "d"]
        assert weighted_kappa(r1, r2, cats, weight="quadratic") == pytest.approx(1.0)

    def test_invalid_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="Weight must be"):
            weighted_kappa(["a"], ["a"], ["a"], weight="cubic")

    def test_single_category(self) -> None:
        """One category → kappa = 1.0."""
        assert weighted_kappa(["a", "a"], ["a", "a"], ["a"]) == 1.0

    def test_different_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            weighted_kappa(["a"], ["a", "b"], ["a", "b"])


class TestConfusionMatrix:
    """Confusion matrix construction."""

    def test_basic_matrix(self) -> None:
        r1 = ["a", "a", "b", "b"]
        r2 = ["a", "b", "a", "b"]
        matrix = confusion_matrix(r1, r2)
        assert matrix["a"]["a"] == 1
        assert matrix["a"]["b"] == 1
        assert matrix["b"]["a"] == 1
        assert matrix["b"]["b"] == 1

    def test_explicit_categories(self) -> None:
        """Categories not in data should still appear."""
        r1 = ["a", "a"]
        r2 = ["a", "a"]
        matrix = confusion_matrix(r1, r2, categories=["a", "b", "c"])
        assert matrix["b"]["b"] == 0
        assert matrix["a"]["a"] == 2

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            confusion_matrix([], [])


class TestFormatConfusionMatrix:
    """Formatting confusion matrix as text."""

    def test_format_output(self) -> None:
        matrix = {"a": {"a": 5, "b": 1}, "b": {"a": 2, "b": 3}}
        text = format_confusion_matrix(matrix, categories=["a", "b"])
        assert "5" in text
        assert "a" in text
        assert "b" in text
