"""LK011: preprocessing fitted on full data, then cross-validated."""

from __future__ import annotations

from collections.abc import Iterator

from ..dataflow import PREPROCESSORS, SELECTORS, Analysis, Role
from ..models import HIGH, Finding, Severity
from ._util import later_cv, later_holdout_split
from .base import Rule, register


@register
class PreprocessingOutsideCV(Rule):
    code = "LK011"
    name = "preprocessing-outside-cv-pipeline"
    summary = "Preprocessing fitted on the full data before cross_val_score / GridSearchCV / a CV loop"
    severity = Severity.ERROR
    confidence = HIGH
    fix = (
        "Wrap preprocessing and model in one Pipeline and pass that to cross_val_score / GridSearchCV: "
        "cross_val_score(make_pipeline(StandardScaler(), model), X, y)."
    )

    def check(self, a: Analysis) -> Iterator[Finding]:
        for fit in a.fits:
            if fit.estimator not in PREPROCESSORS or fit.x_role is not Role.FULL:
                continue  # selectors/samplers before CV are LK007/LK006
            if later_holdout_split(a, fit.line) is not None:
                continue  # LK001
            cv = later_cv(a, fit.line)
            if cv is None:
                continue
            yield self.report(
                a, fit.call.node,
                f"{fit.estimator}.{fit.method}() is fitted on all rows, then cross-validated on line {cv}: "
                "every validation fold was seen by the transformer.",
            )


_ = SELECTORS
