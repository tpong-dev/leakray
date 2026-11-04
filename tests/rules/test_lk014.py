from __future__ import annotations

from leakray import analyze_string
from leakray.models import LOW

HEAD = ("import pandas as pd\nfrom sklearn.model_selection import train_test_split\n"
        'df = pd.read_csv("d.csv")\n')
SPLIT = 'X_tr, X_te, y_tr, y_te = train_test_split(df[["f"]], df["label"])\n'


def _lk014(code: str) -> list:
    return [f for f in analyze_string(HEAD + code + SPLIT) if f.rule == "LK014"]


def test_value_counts_mapped_back_later() -> None:
    assert _lk014('freq = df["c"].value_counts(normalize=True)\ndf["f"] = df["c"].map(freq)\n')


def test_value_counts_eda_only_is_clean() -> None:
    assert not _lk014('freq = df["c"].value_counts()\nprint(freq.head())\n')


def test_groupby_transform_std_on_hint_column_when_target_known() -> None:
    assert _lk014('df["f"] = df.groupby("k")["price"].transform("std")\n')


def test_groupby_rank_assigned_back() -> None:
    assert _lk014('df["f"] = df.groupby("k")["amount"].rank(pct=True)\n')


def test_groupby_agg_merged_back() -> None:
    assert _lk014('s = df.groupby("k")["amount"].agg(["mean"]).reset_index()\ndf = df.merge(s, on="k")\n')


def test_groupby_eda_print_is_clean() -> None:
    assert not _lk014('print(df.groupby("k")["amount"].mean())\n')


def test_full_column_normalisation_is_low() -> None:
    hits = _lk014('df["f"] = df["amount"] / df["amount"].max()\n')
    assert hits and hits[0].confidence == LOW


def test_rowwise_ratio_is_clean() -> None:
    assert not _lk014('df["f"] = df["amount"] / df["items"].clip(lower=1)\n')
