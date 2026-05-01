"""Calibration metrics for binary judge confidence.

Implements Brier score, Expected Calibration Error (ECE), and reliability-diagram
binning from scratch so the math is auditable end-to-end.

ECE binning is *equal-mass* by default (bin by predicted-probability quantile, not
absolute thresholds). Equal-width binning is supported via `binning="equal_width"`
for comparison against standard sklearn-style implementations.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

ProbArray = npt.NDArray[np.floating]
LabelArray = npt.NDArray[np.integer]


def brier_score(probs: ProbArray, labels: LabelArray) -> float:
    """Mean squared error between predicted probabilities and binary labels.

    Brier(p, y) = mean((p - y)^2). Lower is better. Range: [0, 1].

    A perfectly calibrated and perfectly discriminating classifier has Brier 0.
    A constant predictor at base-rate has Brier = p*(1-p) (the irreducible variance
    when predictions carry no information).

    Args:
        probs: predicted probabilities for the positive class, shape (n,).
        labels: ground-truth binary labels in {0, 1}, shape (n,).

    Returns:
        Brier score as a Python float.
    """
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if p.shape != y.shape:
        raise ValueError(f"shape mismatch: probs {p.shape}, labels {y.shape}")
    return float(np.mean((p - y) ** 2))


def _bin_edges(
    probs: ProbArray, n_bins: int, binning: Literal["equal_mass", "equal_width"]
) -> npt.NDArray[np.floating]:
    """Return n_bins+1 edges spanning [0, 1]. Equal-mass uses quantiles of probs."""
    if binning == "equal_width":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if binning == "equal_mass":
        # quantile edges, with explicit 0 and 1 endpoints; deduplicate ties
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(probs, qs)
        edges[0], edges[-1] = 0.0, 1.0
        return np.unique(edges)
    raise ValueError(f"unknown binning: {binning}")


def expected_calibration_error(
    probs: ProbArray,
    labels: LabelArray,
    n_bins: int = 10,
    binning: Literal["equal_mass", "equal_width"] = "equal_mass",
) -> float:
    """Expected Calibration Error: weighted gap between confidence and accuracy.

    ECE = sum_b (|B_b| / n) * |mean_prob(B_b) - mean_acc(B_b)|

    Equal-mass binning (default) makes each bin contain ~n/n_bins examples, so the
    metric is robust to skewed probability distributions. Equal-width uses fixed
    [0, 0.1), [0.1, 0.2), ... bins (sklearn-style).

    Args:
        probs: predicted probabilities, shape (n,).
        labels: ground-truth binary labels, shape (n,).
        n_bins: number of bins (default 10).
        binning: "equal_mass" (default) or "equal_width".

    Returns:
        ECE as a Python float in [0, 1]. Lower is better.
    """
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if p.shape != y.shape:
        raise ValueError(f"shape mismatch: probs {p.shape}, labels {y.shape}")
    edges = _bin_edges(p, n_bins, binning)
    # right-closed bins; clip to ensure 1.0 falls in the last bin
    bin_idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)
    n = len(p)
    ece = 0.0
    for b in range(len(edges) - 1):
        mask = bin_idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        gap = abs(p[mask].mean() - y[mask].mean())
        ece += (count / n) * gap
    return float(ece)


def reliability_diagram_data(
    probs: ProbArray,
    labels: LabelArray,
    n_bins: int = 10,
    binning: Literal["equal_mass", "equal_width"] = "equal_mass",
) -> dict[str, npt.NDArray[np.floating]]:
    """Bin (probs, labels) for plotting a reliability diagram.

    Returns per-bin mean predicted probability (x-axis), mean observed accuracy
    (y-axis), and bin counts (for marker sizing). Empty bins are dropped.

    Args:
        probs: predicted probabilities, shape (n,).
        labels: ground-truth binary labels, shape (n,).
        n_bins: number of bins (default 10).
        binning: "equal_mass" (default) or "equal_width".

    Returns:
        dict with keys:
          - "mean_prob": mean predicted probability per non-empty bin.
          - "accuracy":  mean observed accuracy per non-empty bin.
          - "count":     number of examples per non-empty bin.
    """
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if p.shape != y.shape:
        raise ValueError(f"shape mismatch: probs {p.shape}, labels {y.shape}")
    edges = _bin_edges(p, n_bins, binning)
    bin_idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)
    mean_probs, accs, counts = [], [], []
    for b in range(len(edges) - 1):
        mask = bin_idx == b
        c = int(mask.sum())
        if c == 0:
            continue
        mean_probs.append(p[mask].mean())
        accs.append(y[mask].mean())
        counts.append(c)
    return {
        "mean_prob": np.asarray(mean_probs, dtype=np.float64),
        "accuracy": np.asarray(accs, dtype=np.float64),
        "count": np.asarray(counts, dtype=np.int64),
    }


if __name__ == "__main__":
    # Sanity tests: validate against known closed-form / reference values.
    rng = np.random.default_rng(0)

    # 1. Perfect calibration: p = y in {0,1} => Brier=0, ECE=0.
    y_perf = rng.integers(0, 2, size=1000)
    assert brier_score(y_perf.astype(float), y_perf) == 0.0
    assert expected_calibration_error(y_perf.astype(float), y_perf) == 0.0

    # 2. Constant 0.5 predictor on balanced labels: Brier = 0.25.
    y_bal = rng.integers(0, 2, size=10000)
    p_const = np.full_like(y_bal, 0.5, dtype=float)
    assert abs(brier_score(p_const, y_bal) - 0.25) < 1e-9

    # 3. Pathological over-confidence: p=1 always, half labels correct => Brier=0.5.
    y_half = np.array([0, 1] * 500)
    p_over = np.ones_like(y_half, dtype=float)
    assert abs(brier_score(p_over, y_half) - 0.5) < 1e-9

    # 4. Reliability data: well-calibrated synthetic case has tiny ECE.
    p_sim = rng.uniform(0, 1, size=5000)
    y_sim = (rng.uniform(0, 1, size=5000) < p_sim).astype(int)
    ece = expected_calibration_error(p_sim, y_sim, n_bins=10)
    assert ece < 0.05, f"well-calibrated ECE should be small, got {ece:.4f}"

    rd = reliability_diagram_data(p_sim, y_sim, n_bins=10)
    assert rd["mean_prob"].shape == rd["accuracy"].shape == rd["count"].shape
    assert int(rd["count"].sum()) == 5000

    print("calibration.py sanity tests passed.")
