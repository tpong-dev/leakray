"""LK006: SMOTE / oversampling before the split."""

from __future__ import annotations

import ast
from collections.abc import Iterator

from ..dataflow import SAMPLERS, Analysis, CallInfo, Role
from ..models import HIGH, MEDIUM, Finding, Severity
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
            where = _later(a, fit.line)
            if where is None:
                continue
            via = f" (inside {fit.via_function}())" if fit.via_function else ""
            yield self.report(
                a, fit.call.node,
                f"{fit.estimator}.{fit.method}(){via} runs on the full data before {where}: synthetic samples "
                "built from test rows end up in training, and test rows get near-copies in train.",
            )
        yield from self._manual(a)

    def _manual(self, a: Analysis) -> Iterator[Finding]:
        """sklearn.utils.resample(minority, replace=True) / df.sample(replace=True) before the split."""
        for c in a.calls:
            data, label, conf = _manual_upsample(a, c)
            if data is None or a.role_of(data, c.line) is not Role.FULL:
                continue
            where = _later(a, c.line)
            if where is None:
                continue
            yield self.report(
                a, c.node,
                f"{label} upsamples rows with replacement before {where}: duplicated rows can land in "
                "both train and test, so the test score is inflated.",
                confidence=conf,
            )


def _later(a: Analysis, line: int) -> str | None:
    split = later_holdout_split(a, line)
    if split is not None:
        return f"the split on line {split.line}"
    cv = later_cv(a, line)
    return f"cross-validation on line {cv}" if cv is not None else None


def _manual_upsample(a: Analysis, c: CallInfo) -> tuple[ast.expr | None, str, float]:
    if c.func == "resample" and c.receiver_expr is None and c.args:
        if not (c.name.startswith("sklearn") or c.name == "resample"):
            return None, "", 0.0
        if c.kwarg_const("replace") is False:
            return None, "", 0.0  # downsampling without replacement
        return c.args[0], "resample(..., replace=True)", HIGH if c.name.startswith("sklearn") else MEDIUM
    if c.func == "sample" and c.receiver_expr is not None:
        frac = c.kwarg_const("frac")
        if c.kwarg_const("replace") is True or (isinstance(frac, (int, float)) and frac > 1):
            return c.receiver_expr, ".sample(replace=True)", MEDIUM
    return None, "", 0.0
