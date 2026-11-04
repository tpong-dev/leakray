from __future__ import annotations

from leakray import analyze_string

HEAD = """import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
df = pd.read_csv("d.csv")
"""


def _codes(code: str) -> list[str]:
    return [f.rule for f in analyze_string(HEAD + code)]


def test_scaler_fit_on_test_is_flagged() -> None:
    assert "LK004" in _codes("X_train, X_test = train_test_split(df)\nStandardScaler().fit(X_test)\n")


def test_transform_on_test_is_clean() -> None:
    code = "X_train, X_test = train_test_split(df)\ns = StandardScaler().fit(X_train)\ns.transform(X_test)\n"
    assert "LK004" not in _codes(code)


def test_helper_with_param_named_test_only_transforms() -> None:
    code = (
        "def scale(train, test):\n"
        "    sc = StandardScaler().fit(train)\n"
        "    return sc.transform(train), sc.transform(test)\n"
        "a, b, ya, yb = train_test_split(df[['x']], df['y'])\n"
        "a, b = scale(a, b)\n"
        "Ridge().fit(a, ya).score(b, yb)\n"
    )
    assert "LK004" not in _codes(code)


def test_tuple_reassignment_keeps_sides_apart() -> None:
    code = (
        "train, test = train_test_split(df)\n"
        "m = train.mean()\n"
        "train, test = train.fillna(m), test.fillna(m)\n"
        "Ridge().fit(train[['x']], train['y'])\n"
    )
    assert "LK004" not in _codes(code)


def test_helper_that_fits_called_on_test_is_flagged() -> None:
    code = (
        "def prep(data):\n"
        "    return StandardScaler().fit_transform(data)\n"
        "X_train, X_test = train_test_split(df)\n"
        "X_test = prep(X_test)\n"
    )
    assert "LK004" in _codes(code)
