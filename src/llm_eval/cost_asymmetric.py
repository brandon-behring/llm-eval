"""Cost-asymmetric verifier evaluation utilities.

Companion module to ``notebooks/04_cost_asymmetric_verifier.ipynb``. Implements:

  - synthetic imbalanced binary-classification data,
  - a small PyTorch MLP trained with optional positive-class up-weighting,
  - threshold-aware metrics (precision/recall/F1, precision@k, FP/FN counts),
  - a cost-ratio sweep that picks the empirically optimal threshold for each
    (FP_cost, FN_cost) combination on a held-out probability set.

Math conventions
----------------
For a binary classifier with predicted positive-class probabilities ``p`` and
labels ``y in {0, 1}``, given a decision threshold ``t``:

  - TP = sum(p >= t & y == 1),  FN = sum(p <  t & y == 1)
  - FP = sum(p >= t & y == 0),  TN = sum(p <  t & y == 0)

Total cost at threshold ``t`` for cost matrix ``(c_fp, c_fn)`` is::

    cost(t) = c_fp * FP(t) + c_fn * FN(t)

The optimal threshold ``t*`` minimizes ``cost(t)`` over the empirical set of
candidate thresholds (the unique sorted predicted probabilities). When
``c_fn > c_fp``, ``t*`` shifts *below* 0.5 because flagging more positives is
worth the extra false positives. This is the reverse of the textbook
isotonic-decision rule because we sweep empirically rather than via prior odds.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch import nn

ProbArray = npt.NDArray[np.floating]
LabelArray = npt.NDArray[np.integer]


class SynthSplit(TypedDict):
    """Train/test split returned by :func:`make_imbalanced_synth`."""

    X_train: npt.NDArray[np.floating]
    y_train: npt.NDArray[np.integer]
    X_test: npt.NDArray[np.floating]
    y_test: npt.NDArray[np.integer]


def make_imbalanced_synth(
    n_samples: int = 20000,
    weights: tuple[float, float] = (0.99, 0.01),
    n_features: int = 20,
    n_informative: int = 10,
    test_size: float = 0.2,
    seed: int = 42,
) -> SynthSplit:
    """Synthesize an imbalanced binary-classification dataset and split 80/20.

    Wraps :func:`sklearn.datasets.make_classification`. The default 99/1
    class split simulates a verifier-shaped problem where the rare class is
    the policy violation / fraudulent action that we actually care about.

    Args:
        n_samples: total number of examples generated before splitting.
        weights: class proportions ``(negative, positive)``.
        n_features: total feature dimensionality.
        n_informative: number of informative features (the rest are redundant).
        test_size: fraction held out for the test split.
        seed: random seed for both data generation and splitting.

    Returns:
        Dict with NumPy arrays ``X_train``, ``y_train``, ``X_test``, ``y_test``.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_features - n_informative,
        n_clusters_per_class=2,
        weights=list(weights),
        flip_y=0.01,
        class_sep=1.0,
        random_state=seed,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return {
        "X_train": X_train.astype(np.float32),
        "y_train": y_train.astype(np.int64),
        "X_test": X_test.astype(np.float32),
        "y_test": y_test.astype(np.int64),
    }


class MLP(nn.Module):
    """3-layer MLP for binary classification with a single logit output."""

    def __init__(self, in_features: int, hidden_1: int = 64, hidden_2: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Return logits of shape ``(batch,)``."""
        return self.net(x).squeeze(-1)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_mlp(
    X_train: npt.NDArray[np.floating],
    y_train: npt.NDArray[np.integer],
    pos_weight: float | None = None,
    n_epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
) -> tuple[nn.Module, list[float]]:
    """Train the MLP with BCEWithLogitsLoss; optionally up-weight positives.

    The model is moved to CUDA if available, otherwise stays on CPU. Mini-batch
    indices are shuffled per epoch with a per-epoch RNG seeded from ``seed``.

    Args:
        X_train: training features, shape ``(n, d)``.
        y_train: training labels in ``{0, 1}``, shape ``(n,)``.
        pos_weight: scalar positive-class weight passed to ``BCEWithLogitsLoss``;
            ``None`` means uniform cross-entropy.
        n_epochs: number of full passes through the training data.
        batch_size: mini-batch size.
        lr: Adam learning rate.
        seed: seed for parameter init + minibatch shuffling.

    Returns:
        Tuple of (trained model on CPU, per-epoch mean loss list).
    """
    _set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X = torch.from_numpy(np.asarray(X_train, dtype=np.float32)).to(device)
    y = torch.from_numpy(np.asarray(y_train, dtype=np.float32)).to(device)
    n = X.shape[0]

    model = MLP(in_features=X.shape[1]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    pw = (
        torch.tensor([float(pos_weight)], device=device)
        if pos_weight is not None
        else None
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    rng = np.random.default_rng(seed)
    history: list[float] = []
    for _ in range(n_epochs):
        perm = rng.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        model.train()
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb = X[idx]
            yb = y[idx]
            optim.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optim.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        history.append(epoch_loss / max(n_batches, 1))

    return model.to("cpu"), history


def predict_proba(model: nn.Module, X: npt.NDArray[np.floating]) -> ProbArray:
    """Run the model on CPU and return positive-class probabilities."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(np.asarray(X, dtype=np.float32)))
        probs = torch.sigmoid(logits).numpy()
    return probs.astype(np.float64)


def eval_at_threshold(
    probs: ProbArray, labels: LabelArray, threshold: float
) -> dict[str, float]:
    """Compute confusion-matrix-derived metrics at a fixed decision threshold."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    pred = (p >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def precision_at_k(probs: ProbArray, labels: LabelArray, k: int) -> float:
    """Precision among the top-``k`` items by predicted probability.

    For a verifier ranking decisions by anomaly score, this answers
    "of the ``k`` items I would actually escalate, how many are real?".
    Defaults to 0.0 when ``k <= 0`` or ``k > n``.
    """
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    n = len(p)
    if k <= 0 or k > n:
        return 0.0
    # argpartition gives the top-k indices (unsorted); we just need the labels.
    top_idx = np.argpartition(-p, kth=k - 1)[:k]
    return float(y[top_idx].mean())


def cost_ratio_sweep(
    probs: ProbArray,
    labels: LabelArray,
    fp_costs: list[float],
    fn_costs: list[float],
) -> pd.DataFrame:
    """Find the empirical optimal threshold for each (c_fp, c_fn) pair.

    For each cost combination we sweep over the unique predicted probabilities
    as candidate thresholds (plus 0.0 and 1.0 as endpoints) and pick the
    threshold that minimizes ``c_fp * FP + c_fn * FN``.

    Args:
        probs: predicted positive-class probabilities, shape ``(n,)``.
        labels: ground-truth binary labels, shape ``(n,)``.
        fp_costs: list of false-positive costs to sweep.
        fn_costs: list of false-negative costs to sweep.

    Returns:
        DataFrame with one row per (c_fp, c_fn) pair and columns:
        ``c_fp``, ``c_fn``, ``cost_ratio`` (= c_fn / c_fp),
        ``threshold``, ``total_cost``, ``fp``, ``fn``, ``precision``, ``recall``.
    """
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)

    # Candidate thresholds: unique probs + endpoints. Use ascending order so
    # that ties resolve to the smallest threshold (most aggressive flagging).
    cand = np.unique(np.concatenate([[0.0, 1.0 + 1e-12], p]))

    # Vectorize over candidate thresholds: for each t, compute FP and FN.
    # pred_pos[i, t] = (p_i >= t); shape (n, T).
    # For large n*T this would be heavy; in practice n=4000 test, T<=n, so OK.
    T = cand.size
    # FP(t) = #{p_i >= t & y_i = 0};  FN(t) = #{p_i < t & y_i = 1}.
    # Sort by p descending, then FP(t) = #neg with p_i >= t,
    # FN(t) = #pos - #pos with p_i >= t.
    order = np.argsort(-p)  # descending
    p_sorted = p[order]
    y_sorted = y[order]
    # cumulative counts at each rank position k (top-k flagged):
    #   tp_k = sum_{i<k} y_sorted[i],  fp_k = k - tp_k
    cum_tp = np.cumsum(y_sorted == 1)
    cum_fp = np.cumsum(y_sorted == 0)
    total_pos = int((y == 1).sum())
    # For threshold t, k(t) = #{i: p_sorted[i] >= t}. Use searchsorted on
    # sorted-descending probs by negating.
    # k(t) = number of probs >= t.
    k_at_t = np.searchsorted(-p_sorted, -cand, side="right")
    # Translate k -> (fp, fn). At k=0 (threshold > max prob), fp=0, fn=total_pos.
    fp_at_t = np.where(k_at_t == 0, 0, cum_fp[np.maximum(k_at_t - 1, 0)])
    tp_at_t = np.where(k_at_t == 0, 0, cum_tp[np.maximum(k_at_t - 1, 0)])
    fn_at_t = total_pos - tp_at_t

    rows: list[dict[str, float]] = []
    for c_fp in fp_costs:
        for c_fn in fn_costs:
            costs = c_fp * fp_at_t + c_fn * fn_at_t
            best_idx = int(np.argmin(costs))
            t_star = float(cand[best_idx])
            metrics = eval_at_threshold(p, y, t_star)
            rows.append(
                {
                    "c_fp": float(c_fp),
                    "c_fn": float(c_fn),
                    "cost_ratio": float(c_fn / c_fp),
                    "threshold": t_star,
                    "total_cost": float(costs[best_idx]),
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Sanity tests: end-to-end synth -> train -> evaluate flow.
    data = make_imbalanced_synth(n_samples=4000, seed=42)
    assert data["X_train"].shape == (3200, 20)
    assert data["X_test"].shape == (800, 20)
    pos_rate_train = float(data["y_train"].mean())
    pos_rate_test = float(data["y_test"].mean())
    assert 0.005 < pos_rate_train < 0.02, f"pos rate suspicious: {pos_rate_train}"
    assert 0.005 < pos_rate_test < 0.02, f"pos rate suspicious: {pos_rate_test}"

    # Train a tiny model (10 epochs) just to validate plumbing.
    model, history = train_mlp(
        data["X_train"], data["y_train"], pos_weight=None, n_epochs=10, seed=0
    )
    assert len(history) == 10
    assert all(np.isfinite(h) for h in history), "non-finite training loss"

    probs = predict_proba(model, data["X_test"])
    assert probs.shape == (800,)
    assert (probs >= 0).all() and (probs <= 1).all()

    m05 = eval_at_threshold(probs, data["y_test"], threshold=0.5)
    assert m05["tp"] + m05["fn"] == int(data["y_test"].sum())
    assert m05["fp"] + m05["tn"] == int((data["y_test"] == 0).sum())

    p_at_10 = precision_at_k(probs, data["y_test"], k=10)
    assert 0.0 <= p_at_10 <= 1.0

    sweep = cost_ratio_sweep(
        probs, data["y_test"], fp_costs=[1.0, 10.0], fn_costs=[1.0, 100.0]
    )
    assert sweep.shape == (4, 9)
    # Sanity: when c_fn >> c_fp, optimal threshold should be lower than when
    # c_fn == c_fp (more aggressive flagging is rewarded).
    sym = sweep[(sweep["c_fp"] == 1.0) & (sweep["c_fn"] == 1.0)]["threshold"].iloc[0]
    asym = sweep[(sweep["c_fp"] == 1.0) & (sweep["c_fn"] == 100.0)]["threshold"].iloc[0]
    assert asym <= sym, (
        f"expected lower threshold under high FN cost, got sym={sym:.3f}, "
        f"asym={asym:.3f}"
    )

    print("cost_asymmetric.py sanity tests passed.")
    print(f"  train pos rate = {pos_rate_train:.4f}")
    print(f"  test  pos rate = {pos_rate_test:.4f}")
    print(f"  precision@10  = {p_at_10:.3f}")
    print(f"  t* (sym)      = {sym:.3f}")
    print(f"  t* (FN=100x)  = {asym:.3f}")
