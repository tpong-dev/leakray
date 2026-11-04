from __future__ import annotations

from leakray import analyze_string
from leakray.models import HIGH, MEDIUM

HEAD = "import pandas as pd\nfrom sklearn.model_selection import train_test_split\n"


def _lk008(code: str) -> list:
    return [f for f in analyze_string(HEAD + code) if f.rule == "LK008"]


def test_known_target_beats_name_hints() -> None:
    code = ('df = pd.read_csv("d.csv")\n'
            'df["store_price"] = df.groupby("store")["price"].transform("mean")\n'
            'X_tr, X_te, y_tr, y_te = train_test_split(df[["store_price"]], df["units"])\n')
    assert not _lk008(code)


def test_hint_only_target_is_medium() -> None:
    code = ('df = pd.read_csv("d.csv")\n'
            'df["store_sales"] = df.groupby("store")["sales"].transform("mean")\n'
            'train, test = train_test_split(df)\n')
    hits = _lk008(code)
    assert hits and hits[0].confidence == MEDIUM


def test_map_of_groupby_target_mean_is_high() -> None:
    code = ('df = pd.read_csv("d.csv")\n'
            'df["enc"] = df["cat"].map(df.groupby("cat")["y_col"].mean())\n'
            'X_tr, X_te, y_tr, y_te = train_test_split(df[["enc"]], df["y_col"])\n')
    hits = _lk008(code)
    assert hits and hits[0].confidence == HIGH


def test_helper_function_called_on_full_data() -> None:
    code = ('def add(frame):\n'
            '    frame["r"] = frame.groupby("k")["label"].transform("mean")\n'
            '    return frame\n'
            'df = add(pd.read_csv("d.csv"))\n'
            'X_tr, X_te, y_tr, y_te = train_test_split(df[["r"]], df["label"])\n')
    assert _lk008(code)
