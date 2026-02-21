# Changelog

All notable changes to ir-eval will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-20

### Added

- Core evaluation framework: golden sets, retrieval results, evaluation runs
- **Results-first evaluation**: `ir-eval evaluate results.json --golden golden.json`
- Ranking metrics: Hit Rate, MRR, NDCG@k, Precision@k, MAP
- Agreement metrics: McNemar's test, Fisher exact test
- Confidence metrics: Bootstrap CI, paired permutation test
- Drift detection with 2D severity classification (magnitude x significance)
- Baseline management with version history
- Multiple output formats: console (rich), JSON, Markdown
- CLI with commands: evaluate, run, validate, compare, drift, baseline, history
- Research-KB adapter for live evaluation against research-kb instances
- 142 test functions with hand-calculated known-answer values

### Changed

- Renamed from `llm-eval` to `ir-eval` (classic information retrieval naming)
