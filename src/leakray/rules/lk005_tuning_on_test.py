"""LK005: early stopping / eval_set / hyperparameter tuning on the test set."""

from __future__ import annotations

from collections.abc import Iterator

from ..dataflow import SEARCH_CV, Analysis, Role
from ..models import HIGH, MEDIUM, Finding, Severity
from .base import Rule, register

_EVAL_KWARGS = ("eval_set", "validation_data", "eval_X", "X_val", "valid_sets", "eval_data", "valid_data")
_SCORE_FUNCS = {"predict", "predict_proba", "score", "evaluate", "decision_function"}


@register
class TuningOnTestSet(Rule):
    code = "LK005"
    name = "tuning-on-test-set"
    summary = "Early stopping, eval_set or hyperparameter search uses the test set"
    severity = Severity.ERROR
    confidence = HIGH
    fix = (
        "Carve a validation set out of the training data (train_test_split(X_train, y_train)) and use it "
        "for eval_set / early stopping / tuning; touch the test set once, at the very end."
    )

    def check(self, a: Analysis) -> Iterator[Finding]:
        for c in a.calls:
            for k in _EVAL_KWARGS:
                if k in c.kwargs and a.role_of(c.kwargs[k], c.line) is Role.TEST:
                    yield self.report(
                        a, c.kwargs[k],
                        f"{c.func}(..., {k}=...) uses the test set for early stopping / model selection; "
                        "the reported test score is optimistically biased.",
                    )
        for fit in a.fits:
            if fit.estimator in SEARCH_CV and fit.x_role is Role.TEST:
                yield self.report(a, fit.call.node,
                                  f"{fit.estimator} is tuned on the test set; hyperparameters are chosen on it.")
        # Optuna/hyperopt objective that scores on the test set
        tuned_scopes = {c.func_scope for c in a.calls if c.func.startswith("suggest_") and c.func_scope}
        for c in a.calls:
            if c.func_scope in tuned_scopes and c.func in _SCORE_FUNCS and c.args:
                if a.role_of(c.args[0], c.line) is Role.TEST:
                    yield self.report(
                        a, c.node,
                        f"The tuning objective {c.func_scope}() scores on the test set; tune on a validation split.",
                        confidence=MEDIUM,
                    )
