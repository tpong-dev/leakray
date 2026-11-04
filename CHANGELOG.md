# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- LK002: needs explicit datetime evidence or a column clearly named as a date (no more `spend_per_month` false positives). It also catches shuffled `KFold`/`ShuffleSplit` objects and `cross_val_score(cv=5)` on dated data.
- LK003: resolves `TARGET = "..."` constants, feature-list variables, `list(df.columns)`, `.values`/`iloc[:, :]`, and flags `select_dtypes("number")` at medium confidence.
- LK008 / LK010: a target named in the file (`y = df[c]`, `stratify=`, `fit(X, df[c])`) now beats name hints, which removes false positives on non-target columns. LK008 also follows helper functions called on full data.
- LK012: catches ungrouped splitter objects passed as `cv=`.
- LK014: catches `value_counts(normalize=True)` mapped back, groupby std/rank, aggregates merged back, and whole-column normalisation (low confidence).
- LK013: the test score must be selected inside or right after the loop. Transform and metric variable names (`tf_test`) are no longer treated as data (found by dogfooding).

### Added

- A held-out benchmark set (`benchmark/holdout/`), written after the fixes, to keep the reported numbers honest.

## [0.1.0] - 2026-11-04

### Added

- First release.
- 16 detection rules, LK001-LK016.
- CLI with text, JSON and SARIF output formats.
- `[tool.leakray]` configuration in `pyproject.toml`.
- Inline suppression with `# leakray: ignore[LKxxx]`.
- Jupyter notebook support.
- pre-commit hook.
- GitHub Action with SARIF upload to code scanning.

[Unreleased]: https://github.com/tpong-dev/leakray/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/tpong-dev/leakray/releases/tag/v0.1.0
