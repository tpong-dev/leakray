"""Direct unit tests of the dataflow facts (roles, splits, fits)."""

from __future__ import annotations

from leakray.dataflow import Analysis, Role, combine, guess_estimator_class, role_from_name
from leakray.source import source_from_string

HEAD = """import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("d.csv")
"""
END = 10**6


def _an(code: str) -> Analysis:
    return Analysis(source_from_string(HEAD + code))


def test_combine_and_name_hints() -> None:
    assert combine([Role.TRAIN, Role.TEST, Role.FULL]) is Role.TEST
    assert combine([]) is Role.UNKNOWN
    assert role_from_name("X_test") is Role.TEST
    assert role_from_name("y_val") is Role.VALID
    assert role_from_name("train_df") is Role.TRAIN
    assert role_from_name("test_size") is Role.UNKNOWN
    assert role_from_name("features") is Role.UNKNOWN


def test_loader_is_full_and_derivations_keep_it() -> None:
    a = _an("X = df.drop(columns='y')\n")
    assert a.role_at("df", END) is Role.FULL
    assert a.role_at("X", END) is Role.FULL


def test_train_test_split_roles_and_event() -> None:
    a = _an("a, b, ya, yb = train_test_split(df, df['y'])\n")
    assert [a.role_at(n, END) for n in ("a", "b", "ya", "yb")] == [Role.TRAIN, Role.TEST, Role.TRAIN, Role.TEST]
    (ev,) = a.splits
    assert ev.kind == "train_test_split" and ev.is_random
    assert ev.train == ["a", "ya"] and ev.test == ["b", "yb"]
    assert a.first_split_line == ev.line and a.has_split
    assert a.is_before_any_split(1) and a.split_after(1) is ev and a.split_before(ev.line + 1) is ev


def test_cv_split_loop() -> None:
    a = _an("kf = KFold(5)\nfor tr, te in kf.split(df):\n    pass\n")
    (ev,) = a.splits
    assert ev.kind == "cv_split" and ev.splitter == "KFold"
    assert ev.train == ["tr"] and ev.test == ["te"]


def test_subset_time_slice() -> None:
    a = _an("train = df[df.date < '2020-01-01']\ntest = df[df.date >= '2020-01-01']\n")
    assert [s.kind for s in a.splits] == ["time_slice"]
    assert a.role_at("test", END) is Role.TEST


def test_tuple_assignment_is_elementwise() -> None:
    a = _an("tr, te = train_test_split(df)\ntr, te = tr.fillna(0), te.fillna(0)\n")
    assert a.role_at("tr", END) is Role.TRAIN
    assert a.role_at("te", END) is Role.TEST


def test_call_outputs_follow_positional_inputs() -> None:
    a = _an("tr, te = train_test_split(df)\ntr, te = helper(tr, te)\n")
    assert a.role_at("tr", END) is Role.TRAIN
    assert a.role_at("te", END) is Role.TEST


def test_role_history_is_versioned() -> None:
    a = _an("X = df\ntr, te = train_test_split(X)\nX = te\n")
    split_line = a.splits[0].line
    assert a.role_at("X", split_line) is Role.FULL
    assert a.role_at("X", END) is Role.TEST


def test_fit_with_known_estimator() -> None:
    a = _an("sc = StandardScaler()\nZ = sc.fit_transform(df)\n")
    (fit,) = a.fits
    assert fit.estimator == "StandardScaler" and not fit.estimator_guessed
    assert fit.method == "fit_transform" and fit.x_role is Role.FULL and not fit.has_y


def test_unknown_factory_estimator_is_guessed() -> None:
    a = _an("vec = self._vectorizer()\nZ = vec.fit_transform(df)\n")
    (fit,) = a.fits
    assert fit.estimator == "TfidfVectorizer" and fit.estimator_guessed
    assert fit.role is Role.FULL


def test_guess_from_receiver_name_and_attribute() -> None:
    a = _an("my_scaler.fit(df)\nself.imputer.fit(df)\nmake_pca().fit(df)\n")
    assert [(f.estimator, f.estimator_guessed) for f in a.fits] == [
        ("StandardScaler", True), ("SimpleImputer", True), ("PCA", True)
    ]


def test_data_variable_is_not_guessed() -> None:
    a = _an("vec = df.values\nvec.fit(df)\n")
    assert a.fits[0].estimator is None


def test_guess_table() -> None:
    assert guess_estimator_class("minmax_scaler") == "StandardScaler"
    assert guess_estimator_class("target_encoder") == "TargetEncoder"
    assert guess_estimator_class("ohe") == "OneHotEncoder"
    assert guess_estimator_class("tfidf") == "TfidfVectorizer"
    assert guess_estimator_class("get_vectorizer") == "TfidfVectorizer"
    assert guess_estimator_class("feature_selector") == "SelectKBest"
    assert guess_estimator_class("smote") == "SMOTE"
    assert guess_estimator_class("adasyn") == "ADASYN"
    assert guess_estimator_class("undersampler") == "RandomUnderSampler"
    assert guess_estimator_class("svd") == "TruncatedSVD"
    assert guess_estimator_class("model") is None
    assert guess_estimator_class(None) is None


def test_fit_inside_helper_is_recorded_at_call_site() -> None:
    a = _an(
        "def preprocess(frame):\n"
        "    return StandardScaler().fit_transform(frame)\n"
        "X = preprocess(df)\n"
        "tr, te = train_test_split(X)\n"
    )
    inner, outer = a.fits
    assert inner.via_function is None and inner.x_role is Role.UNKNOWN
    assert outer.via_function == "preprocess" and outer.x_role is Role.FULL
    assert outer.estimator == "StandardScaler" and outer.line > inner.line


def test_helper_kwarg_binding() -> None:
    a = _an("def prep(frame):\n    return StandardScaler().fit_transform(frame)\nprep(frame=df)\n")
    assert a.fits[-1].via_function == "prep" and a.fits[-1].x_role is Role.FULL


def test_helper_without_data_params_adds_nothing() -> None:
    a = _an("def go(k=3):\n    StandardScaler().fit(df)\ngo(4)\n")
    assert len(a.fits) == 1


def test_estimator_alias_and_names() -> None:
    a = _an("sc = StandardScaler()\nm = sc.fit(df)\n")
    assert a.estimator_class("m", END) == "StandardScaler"
    assert "sc" not in a.data_names_in(a.fits[0].call.node)
    assert list(a.calls_named("read_csv"))
