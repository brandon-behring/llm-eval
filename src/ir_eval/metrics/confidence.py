"""Statistical confidence intervals and hypothesis tests.

Pure Python implementation using random + math. Optional scipy acceleration
detected at runtime for Fisher exact test.

Functions:
    - bootstrap_ci: Confidence interval for any statistic
    - paired_bootstrap_test: Compare two systems on same queries
    - fisher_exact_test: 2x2 contingency table exact test
    - mcnemar_test: Paired binary outcomes (hit/miss changes)
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ConfidenceInterval:
    """Bootstrap confidence interval result.

    Args:
        estimate: Point estimate of the statistic.
        lower: Lower bound of the CI.
        upper: Upper bound of the CI.
        confidence_level: Confidence level (e.g., 0.95).
        n_bootstrap: Number of bootstrap samples used.
    """

    estimate: float
    lower: float
    upper: float
    confidence_level: float
    n_bootstrap: int


@dataclass(frozen=True)
class HypothesisTestResult:
    """Result of a hypothesis test.

    Args:
        test_name: Name of the test (e.g., "paired_bootstrap").
        statistic: Test statistic value.
        p_value: Two-sided p-value.
        significant: Whether p < alpha.
        alpha: Significance level used.
        effect_size: Optional effect size (e.g., delta in means).
    """

    test_name: str
    statistic: float
    p_value: float
    significant: bool
    alpha: float = 0.05
    effect_size: float | None = None


def bootstrap_ci(
    values: Sequence[float],
    statistic_fn: Callable[[Sequence[float]], float] | None = None,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> ConfidenceInterval:
    """Compute bootstrap confidence interval for a statistic.

    Works for any metric (MRR, NDCG, Hit Rate) without distributional
    assumptions. Uses the percentile method.

    Args:
        values: Observed values (one per query).
        statistic_fn: Function to compute the statistic from a sample.
            Defaults to arithmetic mean.
        n_bootstrap: Number of bootstrap resamples.
        confidence_level: CI level (e.g., 0.95 for 95% CI).
        seed: Optional random seed for reproducibility.

    Returns:
        ConfidenceInterval with estimate, lower, upper bounds.

    Raises:
        ValueError: If values is empty.
    """
    if not values:
        raise ValueError("Cannot compute CI for empty values")

    if statistic_fn is None:
        def statistic_fn(v: Sequence[float]) -> float:
            return sum(v) / len(v)

    rng = random.Random(seed)
    n = len(values)
    vals = list(values)  # Ensure indexable

    # Point estimate
    estimate = statistic_fn(vals)

    # Bootstrap
    bootstrap_stats: list[float] = []
    for _ in range(n_bootstrap):
        sample = [vals[rng.randint(0, n - 1)] for _ in range(n)]
        bootstrap_stats.append(statistic_fn(sample))

    bootstrap_stats.sort()

    # Percentile method
    alpha = 1 - confidence_level
    lower_idx = max(0, int(math.floor(alpha / 2 * n_bootstrap)) - 1)
    upper_idx = min(n_bootstrap - 1, int(math.ceil((1 - alpha / 2) * n_bootstrap)) - 1)

    return ConfidenceInterval(
        estimate=estimate,
        lower=bootstrap_stats[lower_idx],
        upper=bootstrap_stats[upper_idx],
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
    )


def paired_bootstrap_test(
    values_a: Sequence[float],
    values_b: Sequence[float],
    statistic_fn: Callable[[Sequence[float]], float] | None = None,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> HypothesisTestResult:
    """Paired bootstrap test comparing two systems on the same queries.

    H0: statistic(A) = statistic(B)
    H1: statistic(A) != statistic(B)

    For each bootstrap sample, resample paired indices and compute
    the difference in statistics.

    Args:
        values_a: Per-query metric values from system A.
        values_b: Per-query metric values from system B.
        statistic_fn: Function to aggregate (default: mean).
        n_bootstrap: Number of bootstrap iterations.
        alpha: Significance level.
        seed: Optional random seed.

    Returns:
        HypothesisTestResult with p-value and significance.

    Raises:
        ValueError: If inputs differ in length or are empty.
    """
    if len(values_a) != len(values_b):
        raise ValueError(f"Paired values must have same length: {len(values_a)} vs {len(values_b)}")
    if not values_a:
        raise ValueError("Cannot test empty values")

    if statistic_fn is None:
        def statistic_fn(v: Sequence[float]) -> float:
            return sum(v) / len(v)

    rng = random.Random(seed)
    n = len(values_a)
    a_list = list(values_a)
    b_list = list(values_b)

    # Observed difference
    observed_diff = statistic_fn(b_list) - statistic_fn(a_list)

    # Permutation test under H0: for each pair, randomly swap A/B assignment.
    # Under H0, the two systems are interchangeable, so flipping the label
    # should not change the distribution of the test statistic.
    count_extreme = 0
    for _ in range(n_bootstrap):
        perm_a = []
        perm_b = []
        for i in range(n):
            if rng.random() < 0.5:
                perm_a.append(a_list[i])
                perm_b.append(b_list[i])
            else:
                perm_a.append(b_list[i])
                perm_b.append(a_list[i])
        diff = statistic_fn(perm_b) - statistic_fn(perm_a)

        # Two-sided test
        if abs(diff) >= abs(observed_diff):
            count_extreme += 1

    p_value = count_extreme / n_bootstrap

    return HypothesisTestResult(
        test_name="paired_bootstrap",
        statistic=observed_diff,
        p_value=p_value,
        significant=p_value < alpha,
        alpha=alpha,
        effect_size=observed_diff,
    )


def fisher_exact_test(
    pass_a: int,
    fail_a: int,
    pass_b: int,
    fail_b: int,
    alpha: float = 0.05,
) -> HypothesisTestResult:
    """Fisher's exact test for 2x2 contingency table.

    Appropriate for small sample sizes (n < 50) where chi-squared is unreliable.
    Tests whether the success rates of two systems differ significantly.

    Contingency table:
                System A    System B
        Pass    pass_a      pass_b
        Fail    fail_a      fail_b

    Uses pure Python implementation of the hypergeometric distribution.
    Falls back to scipy.stats.fisher_exact if available.

    Args:
        pass_a: Number of successes for system A.
        fail_a: Number of failures for system A.
        pass_b: Number of successes for system B.
        fail_b: Number of failures for system B.
        alpha: Significance level.

    Returns:
        HypothesisTestResult with two-sided p-value.
    """
    # Try scipy first (more precise)
    try:
        from scipy.stats import fisher_exact as scipy_fisher  # type: ignore[import-untyped]

        table = [[pass_a, pass_b], [fail_a, fail_b]]
        odds_ratio, p_value = scipy_fisher(table, alternative="two-sided")
        return HypothesisTestResult(
            test_name="fisher_exact",
            statistic=odds_ratio,
            p_value=p_value,
            significant=p_value < alpha,
            alpha=alpha,
        )
    except ImportError:
        pass

    # Pure Python fallback using hypergeometric distribution
    # P(X=k) = C(K,k) * C(N-K, n-k) / C(N,n)
    n = pass_a + fail_a + pass_b + fail_b
    row1 = pass_a + pass_b  # total passes
    col1 = pass_a + fail_a  # total system A

    def _log_comb(n: int, k: int) -> float:
        """Log of binomial coefficient C(n, k)."""
        if k < 0 or k > n:
            return float("-inf")
        if k == 0 or k == n:
            return 0.0
        k = min(k, n - k)
        result = 0.0
        for i in range(k):
            result += math.log(n - i) - math.log(i + 1)
        return result

    def _hypergeom_pmf(k: int) -> float:
        """Probability mass function of hypergeometric distribution."""
        log_p = (
            _log_comb(row1, k)
            + _log_comb(n - row1, col1 - k)
            - _log_comb(n, col1)
        )
        return math.exp(log_p)

    # Compute p-value: sum of all probabilities <= P(observed)
    observed_prob = _hypergeom_pmf(pass_a)
    k_min = max(0, col1 - (n - row1))
    k_max = min(col1, row1)

    p_value = 0.0
    for k in range(k_min, k_max + 1):
        prob = _hypergeom_pmf(k)
        if prob <= observed_prob + 1e-10:  # numerical tolerance
            p_value += prob

    # Odds ratio
    if fail_a > 0 and pass_b > 0:
        odds_ratio = (pass_a * fail_b) / (fail_a * pass_b)
    else:
        odds_ratio = float("inf")

    return HypothesisTestResult(
        test_name="fisher_exact",
        statistic=odds_ratio,
        p_value=min(p_value, 1.0),
        significant=p_value < alpha,
        alpha=alpha,
    )


def mcnemar_test(
    both_correct: int,
    a_only: int,
    b_only: int,
    both_wrong: int,
    alpha: float = 0.05,
) -> HypothesisTestResult:
    """McNemar's test for paired binary outcomes.

    Tests whether the marginal proportions of two classifiers/systems
    differ. The key cells are the "discordant" pairs:
    - a_only: A correct, B wrong
    - b_only: B correct, A wrong

    Uses exact binomial test (appropriate for small n).

    Args:
        both_correct: Both systems correct.
        a_only: Only system A correct.
        b_only: Only system B correct.
        both_wrong: Both systems wrong.
        alpha: Significance level.

    Returns:
        HypothesisTestResult with p-value.
    """
    n_discordant = a_only + b_only

    if n_discordant == 0:
        # No discordant pairs → no evidence of difference
        return HypothesisTestResult(
            test_name="mcnemar",
            statistic=0.0,
            p_value=1.0,
            significant=False,
            alpha=alpha,
            effect_size=0.0,
        )

    # Exact binomial test: under H0, P(a_only) = P(b_only) = 0.5
    # Two-sided p-value: 2 * min(P(X <= min_count), P(X >= max_count))
    min_count = min(a_only, b_only)

    # P(X <= min_count) where X ~ Binomial(n_discordant, 0.5)
    p_tail = 0.0
    for k in range(min_count + 1):
        p_tail += math.comb(n_discordant, k) * (0.5 ** n_discordant)

    p_value = min(2.0 * p_tail, 1.0)

    return HypothesisTestResult(
        test_name="mcnemar",
        statistic=float(b_only - a_only),
        p_value=p_value,
        significant=p_value < alpha,
        alpha=alpha,
        effect_size=(b_only - a_only) / n_discordant if n_discordant > 0 else 0.0,
    )
