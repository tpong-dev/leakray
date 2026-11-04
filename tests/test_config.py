from __future__ import annotations

from pathlib import Path

import pytest

from leakray.cli import main
from leakray.config import Config, ConfigError, find_pyproject, load_config
from leakray.models import Severity

LEAKY = """import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("d.csv").drop_duplicates()
X = df.drop(columns=["target"])
y = df["target"]
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y)
"""


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "pyproject.toml"
    p.write_text("[tool.leakray]\n" + body, encoding="utf-8")
    return p


def test_none_path_gives_defaults() -> None:
    assert load_config(None) == Config()


def test_select_string_form_and_extend_ignore(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, 'select = "lk001, LK004,"\nignore = ["LK002"]\n'
                                       'extend_ignore = "lk003"\nmin-confidence = "high"\n'
                                       'exclude = ["build/*"]\nfail-on = "ERROR"\n'))
    assert cfg.select == ["LK001", "LK004"]
    assert cfg.ignore == ["LK002", "LK003"]
    assert cfg.min_confidence >= 0.9
    assert cfg.exclude == ["build/*"]
    assert cfg.fail_on is Severity.ERROR
    assert cfg.source is not None and Path(cfg.source) == tmp_path / "pyproject.toml"


def test_numeric_min_confidence(tmp_path: Path) -> None:
    assert load_config(_write(tmp_path, "min-confidence = 0.5\n")).min_confidence == 0.5


@pytest.mark.parametrize(("body", "msg"), [
    ('min-confidence = "bogus"\n', "min-confidence"),
    ('fail-on = "fatal"\n', "fail-on"),
    ('per-file-ignores = ["LK001"]\n', "per-file-ignores"),
    ("surprise = 1\n", "unknown"),
    ("select = [1, 2]\n", "select"),
    ("select = 5\n", "select"),
])
def test_bad_values_raise(tmp_path: Path, body: str, msg: str) -> None:
    with pytest.raises(ConfigError, match=msg):
        load_config(_write(tmp_path, body))


def test_unreadable_toml(tmp_path: Path) -> None:
    p = tmp_path / "pyproject.toml"
    p.write_text("[tool.leakray\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(p)


def test_per_file_ignores_table(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, '[tool.leakray.per-file-ignores]\n"tests/*" = ["lk001"]\n'))
    assert cfg.per_file_ignores == {"tests/*": ["LK001"]}


def test_find_pyproject_from_file(tmp_path: Path) -> None:
    pp = _write(tmp_path, "")
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    f = sub / "x.py"
    f.write_text("x = 1\n", encoding="utf-8")
    assert find_pyproject(f) == pp.resolve()


def test_cli_per_file_ignores_applied(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = _write(tmp_path, '[tool.leakray.per-file-ignores]\n"nb/*.py" = ["LK"]\n')
    d = tmp_path / "nb"
    d.mkdir()
    (d / "leaky.py").write_text(LEAKY, encoding="utf-8")
    assert main(["check", str(d / "leaky.py"), "--config", str(cfg), "--no-color"]) == 0
    assert "No leakage found" in capsys.readouterr().out


def test_cli_bad_config_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = _write(tmp_path, 'fail-on = "fatal"\n')
    (tmp_path / "x.py").write_text("x = 1\n", encoding="utf-8")
    assert main(["check", str(tmp_path / "x.py"), "--config", str(cfg)]) == 2
    assert "fail-on" in capsys.readouterr().err
