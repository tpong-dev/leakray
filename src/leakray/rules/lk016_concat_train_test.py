"""LK016: train and test concatenated, then used to fit preprocessing."""

from __future__ import annotations

import ast
from collections.abc import Iterator

from ..dataflow import PREPROCESSORS, Analysis
from ..models import HIGH, LOW, MEDIUM, Finding, Severity
from ._util import concat_train_test, is_concat_train_test
from .base import Rule, register

_LABELISH = {"LabelEncoder", "OneHotEncoder", "OrdinalEncoder", "CountVectorizer", "LabelBinarizer"}
_STATS = {"mean", "median", "std", "max", "min", "quantile"}


@register
class ConcatTrainTestBeforeFit(Rule):
    code = "LK016"
    name = "concat-train-test-before-fit"
    summary = "Train and test are concatenated and the combined data is used to fit preprocessing"
    severity = Severity.ERROR
    confidence = HIGH
    fix = "Fit transformers (and compute statistics) on the training data only, then apply them to the test data."

    def check(self, a: Analysis) -> Iterator[Finding]:
        combined = concat_train_test(a)
        for fit in a.fits:
            if not fit.data_args:
                continue
            arg = fit.data_args[0]
            names = a.data_names_in(arg)
            hit = next((n for n in names if n in combined and combined[n] < fit.line), None)
            inline = any(is_concat_train_test(a, n, fit.line) for n in ast.walk(arg))
            if not hit and not inline:
                continue
            if fit.estimator in PREPROCESSORS - _LABELISH:
                conf = HIGH
            elif fit.estimator in _LABELISH:
                conf = MEDIUM  # unsupervised encoders on test features: a milder leak
            else:
                conf = HIGH
            src = f"{hit!r} (train+test concatenated on line {combined[hit]})" if hit else "pd.concat([train, test])"
            yield self.report(a, fit.call.node,
                              f"{fit.estimator or 'Estimator'}.{fit.method}() is fitted on {src}; "
                              "test data shapes the preprocessing.", confidence=conf)
        for c in a.calls_named("fillna"):
            recv = c.receiver
            if recv in combined and c.args and any(
                    isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in _STATS
                    for n in ast.walk(c.args[0])):
                yield self.report(a, c.node, f"Statistics for fillna() are computed on {recv!r}, which contains "
                                  "the test set.", confidence=MEDIUM)
        for c in a.calls_named("get_dummies"):
            if c.args and any(n in combined for n in a.data_names_in(c.args[0])):
                yield self.report(a, c.node, "get_dummies() on train+test combined: test categories define the "
                                  "feature columns (mild leak).", confidence=LOW, severity=Severity.INFO)
