"""Unit tests for LK003 (target column left inside the features)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from leakray import analyze_file
from leakray.models import HIGH, MEDIUM

HEADER = """\
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
df = pd.read_csv("houses.csv")
"""


def _lk003(tmp_path: Path, body: str) -> list:
    p = tmp_path / "case.py"
    p.write_text(HEADER + textwrap.dedent(body), encoding="utf-8")
    res = analyze_file(p)
    assert res.error is None, res.error
    return [f for f in res.findings if f.rule == "LK003"]


LEAKY = {
    "target_constant": 'TARGET = "price"\ny = df[TARGET]\nX = df.drop(columns=["id"])\nRandomForestRegressor().fit(X, y)\n',
    "feature_list_literal": 'features = ["a", "b", "price"]\nX = df[features]\nRandomForestRegressor().fit(X, df["price"])\n',
    "list_of_columns": 'cols = list(df.columns)\nRandomForestRegressor().fit(df[cols], df["price"])\n',
    "list_concat": 'base = ["a"]\nX = df[base + ["price"]]\nRandomForestRegressor().fit(X, df["price"])\n',
    "comprehension_drops_other": 'cols = [c for c in df.columns if c != "id"]\n'
                                 'RandomForestRegressor().fit(df[cols], df["price"])\n',
    "values_whole_frame": 'RandomForestRegressor().fit(df.values, df["price"].values)\n',
    "iloc_whole_frame": 'RandomForestRegressor().fit(df.iloc[:, :], df["price"])\n',
    "loc_with_list": 'RandomForestRegressor().fit(df.loc[:, ["a", "price"]], df["price"])\n',
}

CLEAN = {
    "drop_constant": 'TARGET = "price"\nX = df.drop(TARGET, axis=1)\nRandomForestRegressor().fit(X, df[TARGET])\n',
    "drop_list_var": 'drop_cols = ["id", "price"]\nX = df.drop(columns=drop_cols)\n'
                     'RandomForestRegressor().fit(X, df["price"])\n',
    "comprehension_not_in": 'TARGET = "price"\ncols = [c for c in df.columns if c not in (TARGET, "id")]\n'
                            'RandomForestRegressor().fit(df[cols], df[TARGET])\n',
    "columns_difference": 'cols = df.columns.difference(["price"])\nRandomForestRegressor().fit(df[cols], df["price"])\n',
    "list_remove": 'cols = df.columns.tolist()\ncols.remove("price")\n'
                   'RandomForestRegressor().fit(df[cols], df["price"])\n',
    "inplace_drop_constant": 'TARGET = "price"\ny = df[TARGET]\ndf.drop(columns=[TARGET], inplace=True)\n'
                             'RandomForestRegressor().fit(df, y)\n',
    "unknown_feature_list": 'def run(features):\n    RandomForestRegressor().fit(df[features], df["price"])\n',
    "filtered_comprehension": 'cols = [c for c in df.columns if c.startswith("f_")]\n'
                              'RandomForestRegressor().fit(df[cols], df["price"])\n',
    "select_dtypes_object": 'RandomForestRegressor().fit(df.select_dtypes(include="object"), df["price"])\n',
    "iloc_partial": 'RandomForestRegressor().fit(df.iloc[:, 1:].values, df["price"])\n',
}


@pytest.mark.parametrize("name", sorted(LEAKY))
def test_leaky_patterns_flagged_high(tmp_path: Path, name: str) -> None:
    hits = _lk003(tmp_path, LEAKY[name])
    assert hits, f"{name}: LK003 not reported"
    assert hits[0].confidence >= HIGH
    assert hits[0].extra.get("column") == "price"


@pytest.mark.parametrize("name", sorted(CLEAN))
def test_clean_patterns_not_flagged(tmp_path: Path, name: str) -> None:
    assert not _lk003(tmp_path, CLEAN[name]), f"{name}: unexpected LK003"


@pytest.mark.parametrize("call", ['df.select_dtypes(include="number")', "df.select_dtypes(include=[np.number])",
                                  'df.select_dtypes(exclude=["object"])'])
def test_select_dtypes_is_medium(tmp_path: Path, call: str) -> None:
    hits = _lk003(tmp_path, f'import numpy as np\ny = df["price"]\nX = {call}\nRandomForestRegressor().fit(X, y)\n')
    assert hits
    assert MEDIUM <= hits[0].confidence < HIGH
