"""LK007: supervised feature selection on the full data before the split."""

from __future__ import annotations

import ast
from collections.abc import Iterator

from ..dataflow import SELECTORS, Analysis, Role
from ..models import HIGH, MEDIUM, Finding, Severity
from ._util import ancestors, later_cv, later_holdout_split, subscript_key, target_columns
from .base import Rule, register

_SCORE_FUNCS = {"mutual_info_classif", "mutual_info_regression", "f_classif", "f_regression", "chi2", "r_regression"}


@register
class FeatureSelectionBeforeSplit(Rule):
    code = "LK007"
    name = "feature-selection-before-split"
    summary = "Feature selection that uses y is done on the full data before the split or CV"
    severity = Severity.ERROR
    confidence = HIGH
    fix = (
        "Select features using the training data only, or put the selector inside a Pipeline "
        "(make_pipeline(SelectKBest(k=20), model)) so CV refits it per fold."
    )

    def _later(self, a: Analysis, line: int) -> str | None:
        s = later_holdout_split(a, line)
        if s:
            return f"the split on line {s.line}"
        cv = later_cv(a, line)
        return f"cross-validation on line {cv}" if cv else None

    def check(self, a: Analysis) -> Iterator[Finding]:
        for fit in a.fits:
            if fit.estimator not in SELECTORS or fit.x_role is not Role.FULL:
                continue
            if fit.estimator != "SelectFromModel" and not fit.has_y:
                continue
            later = self._later(a, fit.line)
            if later:
                yield self.report(a, fit.call.node,
                                  f"{fit.estimator}.{fit.method}() picks features with y on the full data before "
                                  f"{later}; the chosen features already 'know' the test labels.")
        for c in a.calls_named(*_SCORE_FUNCS):
            if c.args and a.role_of(c.args[0], c.line) is Role.FULL:
                later = self._later(a, c.line)
                if later:
                    yield self.report(a, c.node, f"{c.func}() scores features against y on the full data before "
                                      f"{later}.", confidence=MEDIUM)
        targets = target_columns(a)
        for c in a.calls_named("corr", "corrwith"):
            if a.role_of(c.receiver_expr, c.line) is not Role.FULL:
                continue
            keys = [subscript_key(p) for p in ancestors(a, c.node) if isinstance(p, ast.Subscript)]
            if any(k in targets for k in keys if k):
                later = self._later(a, c.line)
                if later:
                    yield self.report(a, c.node, f"Features chosen by correlation with the target on the full data "
                                      f"before {later}.", confidence=MEDIUM)
