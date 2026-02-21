"""Inter-rater agreement metrics.

Generalized from interview_practice_bot — works with any set of categories,
not just the 4-level rating scale. Supports weighted Kappa for ordinal scales.

Metrics:
    - cohens_kappa: Chance-corrected agreement
    - weighted_kappa: Ordinal-weighted agreement (linear or quadratic)
    - agreement_rate: Simple percentage agreement
    - confusion_matrix: Category-level agreement breakdown
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence


def confusion_matrix(
    rater1: Sequence[str],
    rater2: Sequence[str],
    categories: Sequence[str] | None = None,
) -> dict[str, dict[str, int]]:
    """Build confusion matrix from two raters' categorical labels.

    Args:
        rater1: Labels from first rater.
        rater2: Labels from second rater.
        categories: Explicit category list (auto-detected if None).

    Returns:
        Nested dict: matrix[rater1_label][rater2_label] = count.

    Raises:
        ValueError: If lists differ in length or are empty.
    """
    if len(rater1) != len(rater2):
        raise ValueError(f"Rater lists must have same length: {len(rater1)} vs {len(rater2)}")
    if len(rater1) == 0:
        raise ValueError("Cannot build confusion matrix for empty lists")

    if categories is None:
        categories = sorted(set(rater1) | set(rater2))

    matrix: dict[str, dict[str, int]] = {
        c1: {c2: 0 for c2 in categories} for c1 in categories
    }
    for label1, label2 in zip(rater1, rater2, strict=False):
        matrix[label1][label2] += 1

    return matrix


def agreement_rate(rater1: Sequence[str], rater2: Sequence[str]) -> float:
    """Compute simple agreement rate (proportion of exact matches).

    Args:
        rater1: Labels from first rater.
        rater2: Labels from second rater.

    Returns:
        Fraction of items where both raters agree (0.0 to 1.0).

    Raises:
        ValueError: If lists differ in length or are empty.
    """
    if len(rater1) != len(rater2):
        raise ValueError(f"Rater lists must have same length: {len(rater1)} vs {len(rater2)}")
    if len(rater1) == 0:
        raise ValueError("Cannot calculate agreement for empty lists")

    matches = sum(1 for a, b in zip(rater1, rater2, strict=False) if a == b)
    return matches / len(rater1)


def cohens_kappa(rater1: Sequence[str], rater2: Sequence[str]) -> float:
    """Compute Cohen's Kappa for nominal categories.

    Kappa = (p_o - p_e) / (1 - p_e)

    where p_o = observed agreement, p_e = expected agreement by chance.

    Interpretation:
        < 0.00: Less than chance
        0.01-0.20: Slight
        0.21-0.40: Fair
        0.41-0.60: Moderate
        0.61-0.80: Substantial
        0.81-1.00: Almost perfect

    Args:
        rater1: Labels from first rater.
        rater2: Labels from second rater.

    Returns:
        Kappa coefficient (-1 to 1).

    Raises:
        ValueError: If lists differ in length or are empty.
    """
    if len(rater1) != len(rater2):
        raise ValueError(f"Rater lists must have same length: {len(rater1)} vs {len(rater2)}")
    if len(rater1) == 0:
        raise ValueError("Cannot calculate kappa for empty lists")

    n = len(rater1)
    categories = sorted(set(rater1) | set(rater2))

    # Observed agreement
    p_o = sum(1 for a, b in zip(rater1, rater2, strict=False) if a == b) / n

    # Expected agreement by chance
    counts1 = Counter(rater1)
    counts2 = Counter(rater2)
    p_e = sum(counts1[cat] * counts2[cat] for cat in categories) / (n * n)

    if p_e == 1.0:
        return 1.0 if p_o == 1.0 else 0.0

    return (p_o - p_e) / (1 - p_e)


def weighted_kappa(
    rater1: Sequence[str],
    rater2: Sequence[str],
    categories: Sequence[str],
    weight: str = "linear",
) -> float:
    """Compute weighted Cohen's Kappa for ordinal categories.

    Weighted Kappa accounts for the magnitude of disagreement. Two raters
    who disagree by one level are penalized less than those who disagree
    by two levels.

    Args:
        rater1: Labels from first rater.
        rater2: Labels from second rater.
        categories: Ordered list of categories (low to high).
        weight: Weight scheme — "linear" or "quadratic".

    Returns:
        Weighted Kappa coefficient (-1 to 1).

    Raises:
        ValueError: If lists differ in length, are empty, or weight is invalid.
    """
    if len(rater1) != len(rater2):
        raise ValueError(f"Rater lists must have same length: {len(rater1)} vs {len(rater2)}")
    if len(rater1) == 0:
        raise ValueError("Cannot calculate weighted kappa for empty lists")
    if weight not in ("linear", "quadratic"):
        raise ValueError(f"Weight must be 'linear' or 'quadratic', got '{weight}'")

    n = len(rater1)
    k = len(categories)
    cat_index = {cat: i for i, cat in enumerate(categories)}

    # Weight matrix
    max_dist = k - 1
    if max_dist == 0:
        return 1.0  # Only one category

    def w(i: int, j: int) -> float:
        dist = abs(i - j)
        if weight == "linear":
            return dist / max_dist
        else:  # quadratic
            return (dist / max_dist) ** 2

    # Observed weighted disagreement
    w_o = 0.0
    for a, b in zip(rater1, rater2, strict=False):
        w_o += w(cat_index[a], cat_index[b])
    w_o /= n

    # Expected weighted disagreement
    counts1 = Counter(rater1)
    counts2 = Counter(rater2)
    w_e = 0.0
    for i, cat_i in enumerate(categories):
        for j, cat_j in enumerate(categories):
            w_e += counts1[cat_i] * counts2[cat_j] * w(i, j)
    w_e /= n * n

    if w_e == 0.0:
        return 1.0

    return 1.0 - (w_o / w_e)


def format_confusion_matrix(
    matrix: dict[str, dict[str, int]],
    categories: Sequence[str] | None = None,
) -> str:
    """Format confusion matrix as a human-readable table.

    Args:
        matrix: Confusion matrix from confusion_matrix().
        categories: Display order (auto-detect if None).

    Returns:
        Formatted string.
    """
    if categories is None:
        categories = sorted(matrix.keys())

    # Column width
    max_cat_len = max(len(c) for c in categories)
    col_width = max(max_cat_len, 6)

    header = " " * (col_width + 2) + "  ".join(f"{c:>{col_width}}" for c in categories)
    lines = [header]
    for row_cat in categories:
        vals = "  ".join(
            f"{matrix[row_cat].get(col_cat, 0):>{col_width}}" for col_cat in categories
        )
        lines.append(f"{row_cat:>{col_width}}  {vals}")

    return "\n".join(lines)
