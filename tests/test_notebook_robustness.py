"""Notebook edge cases found in the field test on public repos."""

from __future__ import annotations

import json
from pathlib import Path

from leakray import analyze_file

LEAK = [
    "import pandas as pd\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler",
    "df = pd.read_csv('d.csv').drop_duplicates()\nX = df.drop(columns=['target'])\ny = df['target']",
    "X = StandardScaler().fit_transform(X)",
    "X_train, X_test, y_train, y_test = train_test_split(X, y)",
]


def _nb(path: Path, cells: list[str], language: str = "python") -> Path:
    nb = {
        "cells": [{"cell_type": "code", "metadata": {}, "source": c, "outputs": [], "execution_count": None}
                  for c in cells],
        "metadata": {"kernelspec": {"name": language, "display_name": language, "language": language}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb), encoding="utf-8")
    return path


def test_broken_cell_costs_only_that_cell(tmp_path: Path) -> None:
    cells = [LEAK[0], "this is not python (", LEAK[1], "%%bash\necho hi", LEAK[2], LEAK[3]]
    res = analyze_file(_nb(tmp_path / "nb.ipynb", cells))
    assert res.error is None
    f = next(f for f in res.findings if f.rule == "LK001")
    assert f.location.cell == 4


def test_r_notebook_is_skipped(tmp_path: Path) -> None:
    res = analyze_file(_nb(tmp_path / "r.ipynb", ["library(tidyverse)\ndf <- read_csv('x.csv')"], language="R"))
    assert res.error is None and res.findings == []


def test_empty_notebook_file_is_skipped(tmp_path: Path) -> None:
    p = tmp_path / "empty.ipynb"
    p.write_text("", encoding="utf-8")
    res = analyze_file(p)
    assert res.error is None and res.findings == []
