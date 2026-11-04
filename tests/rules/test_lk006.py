from __future__ import annotations

from leakray import analyze_string
from leakray.models import HIGH, MEDIUM, Finding

HEAD = """import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.linear_model import LogisticRegression
df = pd.read_csv("d.csv")
X = df.drop(columns="y")
y = df["y"]
"""


def _lk006(code: str) -> list[Finding]:
    return [f for f in analyze_string(HEAD + code) if f.rule == "LK006"]


def test_smote_before_split() -> None:
    assert _lk006("Xr, yr = SMOTE().fit_resample(X, y)\ntr, te = train_test_split(Xr, yr)\n")


def test_smote_on_train_is_clean() -> None:
    code = "X_tr, X_te, y_tr, y_te = train_test_split(X, y)\nXr, yr = SMOTE().fit_resample(X_tr, y_tr)\n"
    assert not _lk006(code)


def test_sampler_inside_helper() -> None:
    code = (
        "def balance(f, l):\n"
        "    return ADASYN().fit_resample(f, l)\n"
        "Xb, yb = balance(X, y)\n"
        "tr, te = train_test_split(Xb, yb)\n"
    )
    hits = _lk006(code)
    assert hits and "balance()" in hits[0].message


def test_guessed_sampler_from_name() -> None:
    code = "sampler = get_sampler()\nXr, yr = sampler.fit_resample(X, y)\ncross_val_score(LogisticRegression(), Xr, yr)\n"
    assert _lk006(code)


def test_sklearn_resample_before_split() -> None:
    code = "mino = df[df.y == 1]\nup = resample(mino, replace=True, n_samples=100)\ntr, te = train_test_split(pd.concat([df, up]))\n"
    hits = _lk006(code)
    assert hits and hits[0].confidence == HIGH


def test_resample_without_replacement_is_clean() -> None:
    code = "down = resample(df, replace=False, n_samples=100)\ntr, te = train_test_split(down)\n"
    assert not _lk006(code)


def test_pandas_sample_with_replacement() -> None:
    hits = _lk006("up = df.sample(frac=2.0)\ntr, te = train_test_split(up)\n")
    assert hits and hits[0].confidence == MEDIUM


def test_pandas_sample_split_is_not_upsampling() -> None:
    assert not _lk006("train = df.sample(frac=0.8)\ntest = df.drop(train.index)\n")


def test_resample_without_later_split_is_clean() -> None:
    assert not _lk006("up = resample(df, replace=True)\n")
