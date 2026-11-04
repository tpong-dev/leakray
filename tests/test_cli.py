from __future__ import annotations

import json
from pathlib import Path

import pytest

from leakray import analyze_string
from leakray.cli import main
from leakray.config import ConfigError, load_config
from leakray.source import load_source

LEAKY = """import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("d.csv").drop_duplicates()
X = df.drop(columns=["target"])
y = df["target"]
X = StandardScaler().fit_transform(X){suffix}
X_train, X_test, y_train, y_test = train_test_split(X, y)
"""


@pytest.fixture
def leaky(tmp_path: Path) -> Path:
    p = tmp_path / "leaky.py"
    p.write_text(LEAKY.format(suffix=""), encoding="utf-8")
    return p


def test_text_output_and_exit_code(leaky: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", str(leaky), "--no-color"]) == 1
    out = capsys.readouterr().out
    assert "LK001" in out and "fix:" in out and "1 finding" in out


def test_exit_zero_and_clean(tmp_path: Path, leaky: Path) -> None:
    assert main(["check", str(leaky), "--exit-zero"]) == 0
    clean = tmp_path / "clean.py"
    clean.write_text("x = 1\n", encoding="utf-8")
    assert main(["check", str(clean)]) == 0


def test_fail_on_error_only(leaky: Path) -> None:
    assert main(["check", str(leaky), "--fail-on", "error"]) == 1
    assert main(["check", str(leaky), "--ignore", "LK001", "--fail-on", "error"]) == 0


def test_select_and_min_confidence(leaky: Path) -> None:
    assert main(["check", str(leaky), "--select", "LK002"]) == 0
    assert main(["check", str(leaky), "--min-confidence", "high"]) == 1


def test_json(leaky: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["check", str(leaky), "--format", "json"])
    data = json.loads(capsys.readouterr().out)
    assert data["summary"]["findings"] == 1
    f = data["findings"][0]
    assert f["rule"] == "LK001" and f["location"]["line"] == 7 and f["fix"]


def test_sarif(leaky: Path, tmp_path: Path) -> None:
    out = tmp_path / "r.sarif"
    main(["check", str(leaky), "--format", "sarif", "--output", str(out)])
    sarif = json.loads(out.read_text(encoding="utf-8"))
    assert sarif["version"] == "2.1.0"
    run = sarif["runs"][0]
    assert run["tool"]["driver"]["name"] == "leakray"
    assert len(run["tool"]["driver"]["rules"]) >= 16
    res = run["results"][0]
    assert res["ruleId"] == "LK001" and res["level"] == "error"
    assert res["locations"][0]["physicalLocation"]["region"]["startLine"] == 7
    assert "partialFingerprints" in res


def test_inline_suppression() -> None:
    assert analyze_string(LEAKY.format(suffix="  # leakray: ignore[LK001]")) == []
    assert analyze_string(LEAKY.format(suffix="  # leakray: ignore")) == []
    assert analyze_string(LEAKY.format(suffix="  # leakray: ignore[LK002]"))
    assert analyze_string("# leakray: ignore-file\n" + LEAKY.format(suffix="")) == []


def test_pyproject_config(tmp_path: Path, leaky: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[tool.leakray]\nignore = ["LK001"]\n', encoding="utf-8")
    assert main(["check", str(leaky)]) == 0
    cfg = load_config(tmp_path / "pyproject.toml")
    assert cfg.ignore == ["LK001"]


def test_bad_config(tmp_path: Path, leaky: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[tool.leakray]\nbogus = 1\n', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(tmp_path / "pyproject.toml")
    assert main(["check", str(leaky)]) == 2


def test_missing_path() -> None:
    assert main(["check", "does/not/exist"]) == 2


def test_rules_and_explain(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["rules"]) == 0
    out = capsys.readouterr().out
    assert "LK001" in out and "LK016" in out
    assert main(["explain", "LK003"]) == 0
    assert "target" in capsys.readouterr().out
    assert main(["explain", "LK999"]) == 2


def test_parse_error_does_not_crash(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "broken.py"
    bad.write_text("def (:\n", encoding="utf-8")
    assert main(["check", str(tmp_path), "--no-color"]) == 0
    assert "could not parse" in capsys.readouterr().out


def test_directory_walk_skips_venv(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "x.py").write_text(LEAKY.format(suffix=""), encoding="utf-8")
    assert main(["check", str(tmp_path)]) == 0


def test_notebook_cell_mapping() -> None:
    nb = Path(__file__).parent / "fixtures" / "LK001" / "bad_notebook.ipynb"
    src = load_source(nb)
    assert src.is_notebook
    from leakray import analyze_file

    f = next(f for f in analyze_file(nb).findings if f.rule == "LK001")
    assert f.location.cell == 3 and f.location.cell_line == 2
