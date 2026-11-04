"""LK004: .fit called on test data."""

from __future__ import annotations

from collections.abc import Iterator

from ..dataflow import SEARCH_CV, Analysis, Role
from ..models import HIGH, Finding, Severity
from ._util import concat_train_test, cv_loop_ranges, is_concat_train_test
from .base import Rule, register


@register
class FitOnTestData(Rule):
    code = "LK004"
    name = "fit-on-test-data"
    summary = "A model or transformer is fitted on the test set"
    severity = Severity.ERROR
    confidence = HIGH
    fix = "Fit only on training data and call .transform()/.predict() on the test set."

    def check(self, a: Analysis) -> Iterator[Finding]:
        concat = concat_train_test(a)
        loops = cv_loop_ranges(a)
        for fit in a.fits:
            if fit.x_role is not Role.TEST or fit.estimator in SEARCH_CV:
                continue  # SearchCV on test is LK005
            arg = fit.data_args[0]
            names = a.data_names_in(arg)
            if any(n in concat for n in names) or any(is_concat_train_test(a, n, fit.line) for n in _walk(arg)):
                continue  # LK016 reports train+test concatenation
            if any(lo <= fit.line <= hi for lo, hi in loops) and not any(
                a.role_at(n, fit.line) is Role.TEST and not _from_cv(a, n) for n in names
            ):
                continue
            yield self.report(
                a, fit.call.node,
                f"{fit.estimator or 'Estimator'}.{fit.method}() is called on test data "
                f"({', '.join(n for n in names if a.role_at(n, fit.line) is Role.TEST) or 'test set'}); "
                "the test set is no longer unseen.",
            )


def _walk(node):  # type: ignore[no-untyped-def]
    import ast

    return list(ast.walk(node))


def _from_cv(a: Analysis, name: str) -> bool:
    for s in a.splits:
        if s.kind == "cv_split" and name in s.test:
            return True
    # X.iloc[test_idx] style: assigned inside the loop from a cv index
    for asg in a.assignments:
        if name in asg.targets:
            for n in a.data_names_in(asg.value):
                if any(s.kind == "cv_split" and n in s.test for s in a.splits):
                    return True
    return False
