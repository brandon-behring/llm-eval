# llm-eval

LLM/judge evaluation lab: calibration, drift detection on judge outputs, and cost-asymmetric verifier evaluation.

> Companion to [ir-eval](https://github.com/brandon-behring/ir-eval) (which evaluates retrieval, not generation).

## Why this exists

[ir-eval](https://github.com/brandon-behring/ir-eval) is a deterministic IR evaluation framework — paired statistical tests on ranking metrics, no LLM-as-judge. For LLM/agent verification systems where the verdict comes from a *judge model*, you need a different evaluation surface:

- **Calibration of judge confidence** — is the judge's `0.7` really a 0.7? (Brier score, ECE, reliability diagrams)
- **Drift detection on judge outputs** — does the judge agree with itself across model versions, customer policies, or temperature changes?
- **Cost-asymmetric evaluation** — when FN cost ≫ FP cost (most policy-violation use cases), which thresholds and metrics matter?
- **Indirect injection robustness** — does the judge prompt withstand attack vectors that don't target the judge itself?

This lab is a sandbox for exploring those problems. Notebooks are runnable end-to-end, with explicit *open question* blocks where I want to learn what production-grade verifier teams do differently.

## Notebooks

| File | Status | Anchors |
|---|---|---|
| `notebooks/01_judge_calibration.ipynb` | planned | Brier + ECE + reliability diagrams on a Claude judge over a labeled subset |
| `notebooks/04_cost_asymmetric_verifier.ipynb` | planned | Small PyTorch classifier with weighted CE loss; FP/FN cost-ratio sweep |
| `notebooks/02_judge_disagreement.ipynb` | future | Cohen's kappa across temperatures; disagreement-mode classification |
| `notebooks/03_composite_drift.ipynb` | future | Multi-metric correlated regression detection on judge outputs (consumes ir-eval) |
| `notebooks/05_indirect_injection.ipynb` | future | Indirect-injection vectors against naive vs hardened judge prompts |

## Setup

```bash
pip install -e ".[dev]"
```

For the judge-calibration notebook you'll also need:
```bash
export ANTHROPIC_API_KEY=...
```

## License

MIT
