# Contributing to leakray

## Dev setup

```bash
py -3.12 -m venv .venv && .venv\Scripts\activate     # Windows; elsewhere: python -m venv .venv
pip install -e ".[dev,docs]"
pytest && ruff check . && mypy
mkdocs serve                                            # preview the docs
```

False positives and missed leaks are the most useful bug reports. Include a minimal snippet.

## How to add a rule

1. **Write the rule module** `src/leakray/rules/lkNNN_slug.py`. Use the next free number. Decorate the class with `@register`, set `code`, `name`, `summary`, `severity`, `confidence` and `fix`, and implement `check(self, a)` as a generator that yields `self.report(a, node, msg)`. Modules named `lk*.py` are discovered automatically; there is no list to edit.
2. **Use the facts in `Analysis`** (roles, splits, fits, calls, assignments), described in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Prefer asking an existing fact over walking the AST yourself.
3. **Add fixtures** in `tests/fixtures/LKNNN/`: every `bad_*.py` must produce the rule's finding and every `good_*.py` must produce none. `tests/test_fixtures.py` runs them automatically. Include a good fixture that looks close to the bad one.
4. **Write the docs page** `docs/rules/LKNNN.md` (copy a neighbour: what it detects, why it leaks, bad, good, when it can be wrong, suppress, references), then add it to `docs/rules/index.md` and the `nav` in `mkdocs.yml`. Update the rule table in `README.md`.
5. **Add benchmark cases** under `benchmark/` with `# expect: LKNNN` labels (and clean cases with no label), then refresh the numbers with `python benchmark/run.py --write`.
6. **Update `CHANGELOG.md`** under `[Unreleased]`.
7. Run `pytest && ruff check . && mypy` and open a pull request.

### Minimal example rule

This example flags `train_test_split` called without `random_state`. It only shows the shape of a rule.

```python
"""LK017: example rule, train_test_split without a random_state."""

from __future__ import annotations

from collections.abc import Iterator

from leakray.dataflow import Analysis
from leakray.models import HIGH, Finding, Severity
from leakray.rules.base import Rule, register


@register
class SplitWithoutSeed(Rule):
    code = "LK017"
    name = "split-without-seed"
    summary = "train_test_split called without random_state"
    severity = Severity.INFO
    confidence = HIGH
    fix = "Pass random_state=<int> so the split is reproducible."

    def check(self, a: Analysis) -> Iterator[Finding]:
        for call in a.calls_named("train_test_split"):
            if call.kwarg_const("random_state") is None:
                yield self.report(
                    a, call.node,
                    "train_test_split() has no random_state; the split changes on every run.",
                )
```

Inside the package, rules use relative imports (`from ..dataflow import Analysis`); see `src/leakray/rules/lk004_fit_on_test.py` for a real one.

Matching fixtures:

```python
# tests/fixtures/LK017/bad_no_seed.py
from sklearn.model_selection import train_test_split
X_tr, X_te = train_test_split(X)

# tests/fixtures/LK017/good_seed.py
from sklearn.model_selection import train_test_split
X_tr, X_te = train_test_split(X, random_state=0)
```

## Style

- Keep messages specific: say which line, which data and what the consequence is.
- Prefer a lower confidence over a false positive. Rules based on name hints should say so in their confidence.
- No new runtime dependencies without discussion.
