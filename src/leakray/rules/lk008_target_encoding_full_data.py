"""LK008: target / mean encoding computed on the full data."""

from __future__ import annotations

import ast
from collections.abc import Iterator

from ..dataflow import TARGET_ENCODERS, Analysis, Role
from ..models import HIGH, MEDIUM, Finding, Severity
from ._util import later_cv, later_holdout_split, target_columns
from .base import Rule, register

_AGG = {"transform", "mean", "agg", "aggregate", "median", "sum"}


def groupby_target_key(a: Analysis, call_receiver: ast.AST | None, targets: dict[str, float]) -> str | None:
    """For ``df.groupby(k)["target"].mean()`` return "target"."""
    if call_receiver is None:
        return None
    for n in ast.walk(call_receiver):
        if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str):
            inner = n.value
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute) and inner.func.attr == "groupby":
                if n.slice.value in targets:
                    return n.slice.value
    return None


@register
class TargetEncodingFullData(Rule):
    code = "LK008"
    name = "target-encoding-on-full-data"
    summary = "Target / mean encoding computed on the full data before the split"
    severity = Severity.ERROR
    confidence = HIGH
    fix = (
        "Compute category means of the target on the training rows only and map them onto the test rows; "
        "better, use sklearn's TargetEncoder (cross-fitted) inside a Pipeline."
    )

    def check(self, a: Analysis) -> Iterator[Finding]:
        for fit in a.fits:
            if fit.estimator in TARGET_ENCODERS and fit.x_role is Role.FULL:
                s = later_holdout_split(a, fit.line)
                cv = later_cv(a, fit.line)
                if s or cv:
                    yield self.report(a, fit.call.node,
                                      f"{fit.estimator}.{fit.method}() learns target means on the full data before "
                                      f"the split on line {s.line if s else cv}; each test row's own label leaks.")
        targets = target_columns(a)
        for c in a.calls:
            if c.func not in _AGG:
                continue
            key = groupby_target_key(a, c.receiver_expr, targets)
            if not key or a.role_of(c.receiver_expr, c.line) is not Role.FULL:
                continue
            s = later_holdout_split(a, c.line)
            if s is None:
                continue
            yield self.report(
                a, c.node,
                f"Mean-of-target encoding: groupby(...)[{key!r}].{c.func}() over the full data before the split on "
                f"line {s.line}; test labels are baked into a feature.",
                confidence=HIGH if targets[key] >= 1.0 else MEDIUM,
            )
