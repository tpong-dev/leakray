"""LK006: SMOTE / oversampling before the split."""

from __future__ import annotations

from collections.abc import Iterator

from ..dataflow import SAMPLERS, Analysis, Role
from ..models import HIGH, Finding, Severity
from ._util import later_cv, later_holdout_split
from .base import Rule, register


@register
class OversamplingBeforeSplit(Rule):
    code = "LK006"
    name = "oversampling-before-split"
    summary = "SMOTE / over- or under-sampling applied to the full data before the split or CV"
    severity = Severity.ERROR
    confidence = HIGH
    fix = (
        "Resample only the training data (X_tr, y_tr = SMOTE().fit_resample(X_train, y_train)), "
        "or use an imblearn Pipeline so CV resamples inside each training fold."
    )

    def check(self, a: Analysis) -> Iterator[Finding]:
        for fit in a.fits:
            if fit.estimator not in SAMPLERS or fit.x_role is not Role.FULL:
                continue
            split = later_holdout_split(a, fit.line)
            cv = later_cv(a, fit.line)
            if split is None and cv is None:
                continue
            where = f"the split on line {split.line}" if split else f"cross-validation on line {cv}"
            yield self.report(
                a, fit.call.node,
                f"{fit.estimator}.{fit.method}() runs on the full data before {where}: synthetic samples "
                "built from test rows end up in training, and test rows get near-copies in train.",
            )
