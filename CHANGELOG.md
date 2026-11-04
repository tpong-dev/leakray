# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
