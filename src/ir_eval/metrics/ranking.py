"""Information retrieval ranking metrics.

Pure Python implementation — no numpy/sklearn dependency.

Metrics:
    - reciprocal_rank: 1/rank of first relevant document
    - hit_at_k: Whether any relevant doc appears in top-k
    - precision_at_k: Fraction of top-k that are relevant
    - average_precision: Area under precision-recall curve
    - ndcg_at_k: Normalized Discounted Cumulative Gain (supports graded relevance)

Aggregate functions:
    - mean_reciprocal_rank (MRR)
    - aggregate_hit_rate
    - mean_average_precision (MAP)
"""

from __future__ import annotations

import math


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Compute reciprocal rank: 1/rank of first relevant document.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs (rank order).
        relevant_ids: Set of known relevant document IDs.

    Returns:
        1/rank of first relevant doc found, or 0.0 if none found.

    Example:
        >>> reciprocal_rank(["a", "b", "c"], {"b", "c"})
        0.5
    """
    for i, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / i
    return 0.0


def hit_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int | None = None) -> bool:
    """Check whether any relevant document appears in top-k results.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs.
        relevant_ids: Set of known relevant document IDs.
        k: Cutoff rank (None = use full list).

    Returns:
        True if at least one relevant doc is in top-k.

    Example:
        >>> hit_at_k(["a", "b", "c"], {"c"}, k=2)
        False
        >>> hit_at_k(["a", "b", "c"], {"c"}, k=3)
        True
    """
    top_k = retrieved_ids[:k] if k is not None else retrieved_ids
    return any(doc_id in relevant_ids for doc_id in top_k)


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int | None = None) -> float:
    """Compute Precision@k: fraction of top-k results that are relevant.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs.
        relevant_ids: Set of known relevant document IDs.
        k: Cutoff rank (None = use full list).

    Returns:
        Number of relevant docs in top-k / k.

    Example:
        >>> precision_at_k(["a", "b", "c", "d"], {"a", "c"}, k=4)
        0.5
    """
    top_k = retrieved_ids[:k] if k is not None else retrieved_ids
    if not top_k:
        return 0.0
    relevant_count = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return relevant_count / len(top_k)


def average_precision(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Compute Average Precision (AP) for a single query.

    AP = (1/R) * sum_{k=1}^{n} P(k) * rel(k)
    where R = total relevant docs, P(k) = precision at cutoff k,
    rel(k) = 1 if doc at rank k is relevant, 0 otherwise.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs.
        relevant_ids: Set of known relevant document IDs.

    Returns:
        Average precision score (0.0 to 1.0).

    Example:
        >>> average_precision(["a", "b", "c"], {"a", "c"})
        0.8333333333333333
    """
    if not relevant_ids:
        return 0.0

    relevant_count = 0
    precision_sum = 0.0

    for i, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            relevant_count += 1
            precision_sum += relevant_count / i

    if relevant_count == 0:
        return 0.0

    return precision_sum / len(relevant_ids)


def _dcg(relevance_scores: list[float], k: int | None = None) -> float:
    """Compute Discounted Cumulative Gain.

    DCG@k = sum_{i=1}^{k} rel_i / log2(i + 1)

    Args:
        relevance_scores: Relevance scores in rank order.
        k: Cutoff (None = use all).

    Returns:
        DCG value.
    """
    scores = relevance_scores[:k] if k is not None else relevance_scores
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(scores))


def ndcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int | None = None,
    relevance_grades: dict[str, int] | None = None,
) -> float:
    """Compute NDCG@k (Normalized Discounted Cumulative Gain).

    Supports both binary and graded relevance:
    - Binary: all docs in relevant_ids get grade 1, others 0.
    - Graded: use relevance_grades dict to assign grades.

    NDCG = DCG / IDCG (Ideal DCG).

    Args:
        retrieved_ids: Ordered list of retrieved document IDs.
        relevant_ids: Set of known relevant document IDs.
        k: Cutoff rank (None = use full list).
        relevance_grades: Optional mapping of doc_id -> grade (higher = more relevant).

    Returns:
        NDCG score (0.0 to 1.0). Returns 0.0 if no relevant docs exist.

    Example:
        >>> ndcg_at_k(["a", "b", "c"], {"a", "c"}, k=3)
        0.8154648767857288
    """
    if not relevant_ids:
        return 0.0

    cutoff = k if k is not None else len(retrieved_ids)
    top_k = retrieved_ids[:cutoff]

    # Build actual relevance vector
    if relevance_grades:
        actual_rels = [float(relevance_grades.get(doc_id, 0)) for doc_id in top_k]
        # Ideal: sort all relevant grades descending
        ideal_rels = sorted(
            [float(relevance_grades.get(doc_id, 0)) for doc_id in relevant_ids],
            reverse=True,
        )
    else:
        # Binary relevance
        actual_rels = [1.0 if doc_id in relevant_ids else 0.0 for doc_id in top_k]
        ideal_rels = [1.0] * len(relevant_ids)

    dcg = _dcg(actual_rels, cutoff)
    idcg = _dcg(ideal_rels, cutoff)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


# ---------------------------------------------------------------------------
# Aggregate functions (operate on lists of per-query values)
# ---------------------------------------------------------------------------


def mean_reciprocal_rank(reciprocal_ranks: list[float]) -> float:
    """Compute Mean Reciprocal Rank (MRR) from per-query reciprocal ranks.

    Args:
        reciprocal_ranks: List of reciprocal rank values.

    Returns:
        Mean of the reciprocal ranks. Returns 0.0 for empty input.

    Example:
        >>> mean_reciprocal_rank([1.0, 0.5, 1/3])
        0.6111111111111112
    """
    if not reciprocal_ranks:
        return 0.0
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def aggregate_hit_rate(hits: list[bool]) -> float:
    """Compute aggregate hit rate from per-query hit results.

    Args:
        hits: List of boolean hit values.

    Returns:
        Fraction of queries with a hit. Returns 0.0 for empty input.

    Example:
        >>> aggregate_hit_rate([True, True, False, True])
        0.75
    """
    if not hits:
        return 0.0
    return sum(hits) / len(hits)


def mean_average_precision(average_precisions: list[float]) -> float:
    """Compute Mean Average Precision (MAP) from per-query AP values.

    Args:
        average_precisions: List of average precision values.

    Returns:
        Mean of the average precisions. Returns 0.0 for empty input.
    """
    if not average_precisions:
        return 0.0
    return sum(average_precisions) / len(average_precisions)
