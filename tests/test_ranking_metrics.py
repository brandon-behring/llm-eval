"""Tests for ranking metrics: MRR, NDCG, Hit Rate, Precision@k, MAP.

Known-value tests verify correctness against hand-calculated results.
"""

from __future__ import annotations

import math

import pytest

from ir_eval.metrics.ranking import (
    aggregate_hit_rate,
    average_precision,
    hit_at_k,
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
)


class TestReciprocalRank:
    """reciprocal_rank: 1/rank of first relevant document."""

    def test_first_position(self) -> None:
        assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0

    def test_second_position(self) -> None:
        assert reciprocal_rank(["x", "a", "c"], {"a"}) == 0.5

    def test_third_position(self) -> None:
        assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)

    def test_not_found(self) -> None:
        assert reciprocal_rank(["x", "y", "z"], {"a"}) == 0.0

    def test_empty_retrieved(self) -> None:
        assert reciprocal_rank([], {"a"}) == 0.0

    def test_multiple_relevant_returns_first(self) -> None:
        """RR uses the FIRST relevant doc, not the best."""
        assert reciprocal_rank(["x", "a", "b"], {"a", "b"}) == 0.5

    def test_empty_relevant(self) -> None:
        assert reciprocal_rank(["a", "b"], set()) == 0.0


class TestHitAtK:
    """hit_at_k: whether any relevant doc appears in top-k."""

    def test_hit_within_k(self) -> None:
        assert hit_at_k(["x", "a", "y"], {"a"}, k=3) is True

    def test_miss_outside_k(self) -> None:
        assert hit_at_k(["x", "y", "a"], {"a"}, k=2) is False

    def test_hit_at_boundary(self) -> None:
        assert hit_at_k(["x", "a"], {"a"}, k=2) is True

    def test_no_cutoff(self) -> None:
        assert hit_at_k(["x", "y", "z", "a"], {"a"}) is True

    def test_empty_results(self) -> None:
        assert hit_at_k([], {"a"}, k=5) is False


class TestPrecisionAtK:
    """precision_at_k: fraction of top-k that are relevant."""

    def test_perfect_precision(self) -> None:
        assert precision_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0

    def test_half_precision(self) -> None:
        assert precision_at_k(["a", "x", "b", "y"], {"a", "b"}, k=4) == 0.5

    def test_zero_precision(self) -> None:
        assert precision_at_k(["x", "y", "z"], {"a"}, k=3) == 0.0

    def test_no_cutoff(self) -> None:
        assert precision_at_k(["a", "x"], {"a"}) == 0.5

    def test_empty_results(self) -> None:
        assert precision_at_k([], {"a"}, k=5) == 0.0


class TestAveragePrecision:
    """average_precision: area under precision-recall curve."""

    def test_perfect_retrieval(self) -> None:
        """Both relevant docs at positions 1 and 2."""
        ap = average_precision(["a", "b", "x"], {"a", "b"})
        # P(1)*1 + P(2)*1 = (1/1 + 2/2) / 2 = 1.0
        assert ap == pytest.approx(1.0)

    def test_mixed_positions(self) -> None:
        """Relevant docs at positions 1 and 3."""
        ap = average_precision(["a", "x", "b"], {"a", "b"})
        # P(1)*1 + P(3)*1 = (1/1 + 2/3) / 2 = 0.8333
        assert ap == pytest.approx(5 / 6)

    def test_no_relevant_found(self) -> None:
        assert average_precision(["x", "y", "z"], {"a"}) == 0.0

    def test_empty_relevant_set(self) -> None:
        assert average_precision(["a", "b"], set()) == 0.0

    def test_single_relevant_at_first(self) -> None:
        ap = average_precision(["a", "x", "y"], {"a"})
        assert ap == pytest.approx(1.0)

    def test_single_relevant_at_third(self) -> None:
        ap = average_precision(["x", "y", "a"], {"a"})
        assert ap == pytest.approx(1 / 3)


class TestNdcgAtK:
    """ndcg_at_k: Normalized Discounted Cumulative Gain."""

    def test_perfect_ranking(self) -> None:
        """Single relevant doc at rank 1 → NDCG = 1.0."""
        assert ndcg_at_k(["a", "x", "y"], {"a"}, k=3) == pytest.approx(1.0)

    def test_relevant_at_rank_2(self) -> None:
        """Single relevant doc at rank 2.
        DCG = 1/log2(3) = 0.6309
        IDCG = 1/log2(2) = 1.0
        NDCG = 0.6309
        """
        expected = (1 / math.log2(3)) / (1 / math.log2(2))
        assert ndcg_at_k(["x", "a", "y"], {"a"}, k=3) == pytest.approx(expected)

    def test_no_relevant_found(self) -> None:
        assert ndcg_at_k(["x", "y", "z"], {"a"}, k=3) == 0.0

    def test_empty_relevant_set(self) -> None:
        assert ndcg_at_k(["a", "b"], set(), k=3) == 0.0

    def test_all_relevant(self) -> None:
        """All docs relevant → NDCG = 1.0."""
        assert ndcg_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == pytest.approx(1.0)

    def test_graded_relevance(self) -> None:
        """Graded relevance: high-grade doc at rank 1 is better."""
        grades = {"a": 3, "b": 1}
        # DCG = 3/log2(2) + 1/log2(3) = 3.0 + 0.6309 = 3.6309
        # IDCG = 3/log2(2) + 1/log2(3) = 3.6309 (already in ideal order)
        ndcg = ndcg_at_k(["a", "b"], {"a", "b"}, k=2, relevance_grades=grades)
        assert ndcg == pytest.approx(1.0)

    def test_graded_relevance_suboptimal(self) -> None:
        """Low-grade doc first → NDCG < 1.0."""
        grades = {"a": 3, "b": 1}
        ndcg = ndcg_at_k(["b", "a"], {"a", "b"}, k=2, relevance_grades=grades)
        # DCG = 1/log2(2) + 3/log2(3) = 1.0 + 1.8927 = 2.8927
        # IDCG = 3/log2(2) + 1/log2(3) = 3.0 + 0.6309 = 3.6309
        expected = (1 / math.log2(2) + 3 / math.log2(3)) / (3 / math.log2(2) + 1 / math.log2(3))
        assert ndcg == pytest.approx(expected)

    def test_no_cutoff(self) -> None:
        """k=None uses full list."""
        ndcg = ndcg_at_k(["a", "x"], {"a"})
        assert ndcg == pytest.approx(1.0)


class TestMeanReciprocalRank:
    """MRR: aggregate over multiple queries."""

    def test_known_mrr(self) -> None:
        """MRR([1, 1/2, 1/3]) = 0.6111."""
        mrr = mean_reciprocal_rank([1.0, 0.5, 1 / 3])
        assert mrr == pytest.approx(11 / 18)

    def test_empty(self) -> None:
        assert mean_reciprocal_rank([]) == 0.0

    def test_all_perfect(self) -> None:
        assert mean_reciprocal_rank([1.0, 1.0, 1.0]) == 1.0

    def test_all_miss(self) -> None:
        assert mean_reciprocal_rank([0.0, 0.0]) == 0.0


class TestAggregateHitRate:
    """Aggregate hit rate over multiple queries."""

    def test_known_rate(self) -> None:
        assert aggregate_hit_rate([True, True, False, True]) == 0.75

    def test_empty(self) -> None:
        assert aggregate_hit_rate([]) == 0.0

    def test_all_hits(self) -> None:
        assert aggregate_hit_rate([True, True, True]) == 1.0


class TestMeanAveragePrecision:
    """MAP: aggregate over multiple queries."""

    def test_known_map(self) -> None:
        map_val = mean_average_precision([1.0, 5 / 6, 1 / 3])
        assert map_val == pytest.approx((1.0 + 5 / 6 + 1 / 3) / 3)

    def test_empty(self) -> None:
        assert mean_average_precision([]) == 0.0
