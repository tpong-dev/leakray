from __future__ import annotations

from leakray import analyze_string

HEAD = "import pandas as pd\nfrom sklearn.model_selection import train_test_split\n"


def _lk010(code: str) -> list:
    return [f for f in analyze_string(HEAD + code) if f.rule == "LK010"]


def test_diff_on_hint_named_feature_is_clean_when_target_known() -> None:
    code = ('df = pd.read_csv("d.csv")\n'
            'df["price_change"] = df["price"].diff()\n'
            'y = df["demand"]\n')
    assert not _lk010(code)


def test_diff_on_known_target_is_flagged() -> None:
    code = ('df = pd.read_csv("d.csv")\n'
            'df["d"] = df["demand"].diff()\n'
            'y = df["demand"]\n')
    assert _lk010(code)


def test_assign_lambda_rolling_on_target() -> None:
    code = ('df = pd.read_csv("d.csv")\n'
            'df = df.assign(ma=lambda d: d["demand"].rolling(3).mean())\n'
            'y = df["demand"]\n')
    assert _lk010(code)
