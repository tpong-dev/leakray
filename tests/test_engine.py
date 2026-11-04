from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from leakray import engine
from leakray.engine import Options, analyze_file, analyze_string, parse_error_finding
from leakray.models import Finding, Location, Severity
from leakray.source import load_source, notebook_code, source_from_string

LEAKY = """import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("d.csv").drop_duplicates()
X = df.drop(columns=["target"])
y = df["target"]
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y)
"""


def _codes(findings: list[Finding]) -> set[str]:
    return {f.rule for f in findings}


def test_select_prefix_runs_matching_rules() -> None:
    assert "LK001" in _codes(analyze_string(LEAKY, Options(select=["LK00"])))
    assert "LK001" in _codes(analyze_string(LEAKY, Options(select=["lk001"])))
    assert "LK001" not in _codes(analyze_string(LEAKY, Options(select=["LK01"])))
    assert "LK001" not in _codes(analyze_string(LEAKY, Options(ignore=["LK00"])))


def test_min_confidence_filters_everything() -> None:
    assert analyze_string(LEAKY, Options(min_confidence=1.01)) == []


def test_ignore_file_with_specific_codes() -> None:
    other = "# leakray: ignore-file[LK999]\n" + LEAKY
    assert "LK001" in _codes(analyze_string(other))
    res = engine.analyze_source(source_from_string("# leakray: ignore-file[LK001, LK002]\n" + LEAKY))
    assert "LK001" not in _codes(res.findings)
    assert res.suppressed >= 1


def test_ignore_file_all_codes() -> None:
    res = engine.analyze_source(source_from_string("# leakray: ignore-file\n" + LEAKY))
    assert res.findings == [] and res.suppressed >= 1


def test_suppression_on_later_line_of_multiline_call() -> None:
    one = "X = StandardScaler().fit_transform(X)\n"
    base = LEAKY.replace(one, "X = StandardScaler().fit_transform(\n    X,\n)\n")
    multi = LEAKY.replace(one, "X = StandardScaler().fit_transform(\n    X,  # leakray: ignore[LK001]\n)\n")
    assert "LK001" in _codes(analyze_string(base))
    res = engine.analyze_source(source_from_string(multi))
    assert "LK001" not in _codes(res.findings)
    assert res.suppressed >= 1


def test_line_suppression_wrong_code_keeps_finding() -> None:
    code = LEAKY.replace("fit_transform(X)\n", "fit_transform(X)  # leakray: ignore[LK004]\n")
    assert "LK001" in _codes(analyze_string(code))


class _Boom:
    code = "LK901"

    def check(self, a: object) -> Iterator[Finding]:
        raise RuntimeError("boom")


class _Dup:
    code = "LK902"

    def check(self, a: object) -> Iterator[Finding]:
        f = Finding(rule="LK902", name="dup", message="m", severity=Severity.INFO, confidence=1.0,
                    location=Location(path="<string>", line=1), fix="x")
        yield f
        yield f


def test_buggy_rule_does_not_crash_and_duplicates_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine, "all_rules", lambda: [_Boom, _Dup])
    res = engine.analyze_source(source_from_string("x = 1\n"))
    assert res.error is not None and "LK901" in res.error and "boom" in res.error
    assert [f.rule for f in res.findings] == ["LK902"]


def test_analyze_file_parse_error(tmp_path: Path) -> None:
    p = tmp_path / "bad.py"
    p.write_text("def (:\n", encoding="utf-8")
    res = analyze_file(p)
    assert res.findings == []
    assert res.error is not None and res.error.startswith("could not parse")
    assert Path(res.path) == p


def test_parse_error_finding() -> None:
    f = parse_error_finding("a.py", "bad")
    assert f.rule == "LK000" and f.location.line == 1 and f.severity is Severity.INFO


def _cell(ctype: str, src: str | list[str]) -> dict[str, object]:
    c: dict[str, object] = {"cell_type": ctype, "metadata": {}, "source": src}
    if ctype == "code":
        c.update(outputs=[], execution_count=None)
    return c


def _nb(cells: list[dict[str, object]]) -> str:
    return json.dumps({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5})


def test_notebook_bash_and_syntax_error_cells_parse_leniently(tmp_path: Path) -> None:
    nb = _nb([
        _cell("markdown", "# title"),
        _cell("code", "%%bash\necho hi\nls -la"),
        _cell("code", "%%time\nx = 1"),
        _cell("code", "def broken(:\n    pass"),
        _cell("code", "!pip install foo\n%matplotlib inline\nz = 3"),
        _cell("code", LEAKY.splitlines(keepends=True)),
    ])
    p = tmp_path / "nb.ipynb"
    p.write_text(nb, encoding="utf-8")
    src = load_source(p)
    assert src.is_notebook
    assert "# %%bash" in src.text and "pass  # %%time" in src.text
    assert src.line_text(0) == "" and src.line_text(10_000) == ""
    assert src.map_line(0) == (None, None)
    res = analyze_file(p)
    assert res.error is None
    lk1 = [f for f in res.findings if f.rule == "LK001"]
    assert lk1 and lk1[0].location.cell == 5 and lk1[0].location.cell_line == 7


def test_notebook_code_falls_back_to_plain_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "nbformat", None)  # import fails -> json fallback
    raw = _nb([_cell("code", ["a = 1\n", "b = 2"]), _cell("markdown", "hi"), _cell("code", "")])
    text, cell_map = notebook_code(raw)
    assert text == "a = 1\nb = 2\n\n"
    assert cell_map == [(0, 1), (0, 2), (2, 1)]


def test_plain_source_map_line_is_none() -> None:
    assert source_from_string("x = 1\n").map_line(1) == (None, None)
