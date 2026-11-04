from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from leakray.cli import main
from leakray.engine import FileResult, analyze_file
from leakray.output import render_json, render_text, summary, to_sarif

LEAKY = """import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("d.csv").drop_duplicates()
X = df.drop(columns=["target"])
y = df["target"]
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y)
"""


@pytest.fixture
def leaky(tmp_path: Path) -> Path:
    p = tmp_path / "leaky.py"
    p.write_text(LEAKY, encoding="utf-8")
    return p


@pytest.fixture
def notebook(tmp_path: Path) -> Path:
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": "# intro"},
        {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
         "source": LEAKY.splitlines(keepends=True)},
    ]
    p = tmp_path / "leaky.ipynb"
    p.write_text(json.dumps({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}),
                 encoding="utf-8")
    return p


@pytest.fixture
def bad(tmp_path: Path) -> Path:
    p = tmp_path / "bad.py"
    p.write_text("def (:\n", encoding="utf-8")
    return p


def _text(results: list[FileResult], quiet: bool = False, statistics: bool = False) -> str:
    buf = io.StringIO()
    render_text(results, buf, color=False, quiet=quiet, statistics=statistics)
    return buf.getvalue()


def test_text_statistics_and_quiet(leaky: Path) -> None:
    res = [analyze_file(leaky)]
    out = _text(res, quiet=True, statistics=True)
    n = sum(f.rule == "LK001" for f in res[0].findings)
    assert n >= 1
    assert f"{n:>5}  LK001" in out
    assert "fix:" not in out and "docs:" not in out and " | " not in out
    full = _text(res)
    assert "fix:" in full and "docs:" in full and "   7 |" in full
    assert "LK001" not in full.split("finding(s)")[0].splitlines()[-1]


def test_text_statistics_omitted_when_clean(tmp_path: Path) -> None:
    p = tmp_path / "clean.py"
    p.write_text("x = 1\n", encoding="utf-8")
    out = _text([analyze_file(p)], statistics=True)
    assert "No leakage found" in out and "LK0" not in out


def test_text_notebook_location(notebook: Path) -> None:
    out = _text([analyze_file(notebook)])
    assert f"{notebook} cell 2, line 7:" in out
    assert "   7 |" in out


def test_text_parse_error_warning(bad: Path) -> None:
    out = _text([analyze_file(bad)])
    assert "warning:" in out and "could not parse" in out


def test_sarif_notebook_region_and_prefix(notebook: Path) -> None:
    sarif = to_sarif([analyze_file(notebook)])
    results = [r for r in sarif["runs"][0]["results"] if r["ruleId"] == "LK001"]
    assert results
    r = results[0]
    assert r["locations"][0]["physicalLocation"]["region"] == {"startLine": 1}
    assert r["message"]["text"].startswith("[cell 2, line 7] ")


def test_sarif_rule_index_and_fingerprints_stable(leaky: Path) -> None:
    a = to_sarif([analyze_file(leaky)])
    b = to_sarif([analyze_file(leaky)])
    rules = a["runs"][0]["tool"]["driver"]["rules"]
    ra, rb = a["runs"][0]["results"], b["runs"][0]["results"]
    assert ra and len(ra) == len(rb)
    for x, y in zip(ra, rb):
        assert x["partialFingerprints"] == y["partialFingerprints"]
        assert len(x["partialFingerprints"]["leakray/v1"]) == 32
        assert rules[x["ruleIndex"]]["id"] == x["ruleId"]
        assert x["ruleIndex"] == y["ruleIndex"]
    lk1 = next(x for x in ra if x["ruleId"] == "LK001")
    loc = lk1["locations"][0]["physicalLocation"]
    assert loc["region"]["startLine"] == 7 and loc["region"]["startColumn"] >= 1
    assert "\\" not in loc["artifactLocation"]["uri"]


def test_json_errors_list(bad: Path, leaky: Path) -> None:
    buf = io.StringIO()
    results = [analyze_file(leaky), analyze_file(bad)]
    render_json(results, buf)
    data = json.loads(buf.getvalue())
    assert len(data["errors"]) == 1
    assert Path(data["errors"][0]["path"]) == bad
    assert "could not parse" in data["errors"][0]["error"]
    assert data["summary"]["parse_errors"] == 1 == summary(results)["parse_errors"]
    assert data["summary"]["files"] == 2


def test_cli_json_warns_on_stderr(bad: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", str(bad), "--format", "json"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["errors"]
    assert "leakray: warning:" in captured.err


def test_cli_statistics_quiet(leaky: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", str(leaky), "--no-color", "--quiet", "--statistics"]) == 1
    out = capsys.readouterr().out
    assert "LK001" in out and "fix:" not in out


def test_cli_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "usage" in capsys.readouterr().out.lower()
