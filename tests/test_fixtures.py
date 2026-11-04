"""Every rule ships bad/good fixtures under tests/fixtures/LKxxx/.

* ``bad_*.py`` / ``bad_*.ipynb`` must produce at least one finding of that rule.
* ``good_*.py`` / ``good_*.ipynb`` must produce no finding of that rule.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from leakray import analyze_file
from leakray.rules import all_rules

FIXTURES = Path(__file__).parent / "fixtures"


def _cases(prefix: str) -> list[Path]:
    if not FIXTURES.exists():
        return []
    return sorted(p for p in FIXTURES.glob(f"LK*/{prefix}_*") if p.suffix in {".py", ".ipynb"})


@pytest.mark.parametrize("path", _cases("bad"), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_bad_fixture_is_flagged(path: Path) -> None:
    code = path.parent.name
    res = analyze_file(path)
    assert res.error is None, res.error
    codes = [f.rule for f in res.findings]
    assert code in codes, f"{code} not reported for {path.name}; got {codes}"


@pytest.mark.parametrize("path", _cases("good"), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_good_fixture_is_clean(path: Path) -> None:
    code = path.parent.name
    res = analyze_file(path)
    assert res.error is None, res.error
    hits = [f for f in res.findings if f.rule == code]
    assert not hits, f"false positive {code} in {path.name}: {[str(f.location) for f in hits]}"


def test_every_rule_has_fixtures() -> None:
    for rule in all_rules():
        d = FIXTURES / rule.code
        assert list(d.glob("bad_*")), f"{rule.code} has no bad fixture"
        assert list(d.glob("good_*")), f"{rule.code} has no good fixture"


def test_every_rule_has_metadata() -> None:
    codes = set()
    for rule in all_rules():
        assert rule.code.startswith("LK") and len(rule.code) == 5
        assert rule.name and rule.summary and rule.fix
        assert rule.code not in codes
        codes.add(rule.code)
