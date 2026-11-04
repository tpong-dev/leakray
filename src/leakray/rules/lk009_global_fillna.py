"""LK009: fillna with a global statistic before the split."""

from __future__ import annotations

import ast
from collections.abc import Iterator

from ..dataflow import Analysis, Role
from ..models import MEDIUM, Finding, Severity
from ._util import later_holdout_split
from .base import Rule, register

_STATS = {"mean", "median", "mode", "std", "quantile", "max", "min"}


def _stat_call(node: ast.AST) -> ast.Call | None:
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in _STATS:
            return n
    return None


@register
class GlobalFillnaBeforeSplit(Rule):
    code = "LK009"
    name = "global-fillna-before-split"
    summary = "Missing values filled with a statistic of the full data before the split"
    severity = Severity.WARNING
    confidence = MEDIUM
    fix = (
        "Compute the statistic on the training data and reuse it: X_test.fillna(X_train.mean()); "
        "or use SimpleImputer inside a Pipeline."
    )

    def check(self, a: Analysis) -> Iterator[Finding]:
        for c in a.calls_named("fillna", "replace"):
            value = c.args[-1] if c.args else c.kwargs.get("value")
            if value is None:
                continue
            stat = _stat_call(value)
            if stat is None:
                continue
            if a.role_of(c.receiver_expr, c.line) is not Role.FULL:
                continue
            if a.role_of(stat.func.value, c.line) not in (Role.FULL, Role.UNKNOWN):  # type: ignore[attr-defined]
                continue
            s = later_holdout_split(a, c.line)
            if s is None:
                continue
            yield self.report(
                a, c.node,
                f"{c.func}() fills gaps with .{stat.func.attr}() of the full data before the split on "  # type: ignore[attr-defined]
                f"line {s.line}; test values shape the training features.",
            )
