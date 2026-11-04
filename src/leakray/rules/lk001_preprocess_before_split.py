"""LK001: scaler / imputer / encoder / PCA fitted on the full data before the split."""

from __future__ import annotations

from collections.abc import Iterator

from ..dataflow import PREPROCESSORS, Analysis, Role
from ..models import HIGH, Finding, Severity
from .base import Rule, register


@register
class PreprocessBeforeSplit(Rule):
    code = "LK001"
    name = "preprocessing-fit-before-split"
    summary = "Scaler, imputer, encoder or PCA fitted on the full dataset before the train/test split"
    severity = Severity.ERROR
    confidence = HIGH
    fix = (
        "Split first, then fit the transformer on the training data only "
        "(scaler.fit(X_train); scaler.transform(X_test)), or put it inside a sklearn Pipeline."
    )

    def check(self, a: Analysis) -> Iterator[Finding]:
        for fit in a.fits:
            if fit.estimator not in PREPROCESSORS or fit.method not in {"fit", "fit_transform"}:
                continue
            if fit.x_role is not Role.FULL:
                continue
            later = a.split_after(fit.line)
            if later is None:
                continue  # no later split: CV case is LK011, no-split files are out of scope
            yield self.report(
                a,
                fit.call.node,
                f"{fit.estimator}.{fit.method}() is fitted on the full dataset before the "
                f"train/test split on line {later.line}; test-set statistics leak into training.",
            )
