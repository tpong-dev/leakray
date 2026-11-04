"""LK008: target / mean encoding computed on the full data."""

from __future__ import annotations

import ast
from collections.abc import Iterator

from ..dataflow import TARGET_ENCODERS, Analysis, CallInfo, Role, SplitEvent
from ..models import HIGH, MEDIUM, Finding, Severity
from ._util import later_cv, later_holdout_split, resolved_target_columns
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


def _full_data_call_site(a: Analysis, func: str) -> int | None:
    """Line of the first call ``func(<full data>)`` at module level, for helpers like ``add_rate(df)``."""
    for c in a.calls:
        if c.func_scope is not None or c.func != func or not isinstance(c.node.func, ast.Name):
            continue
        if any(a.role_of(arg, c.line) is Role.FULL for arg in c.args):
            return c.line
        # sales = add_rate(pd.read_csv(...)) -> the assigned name is FULL
        for asg in a.assignments:
            if asg.line == c.line and asg.role is Role.FULL:
                return c.line
    return None


def full_data_context(a: Analysis, c: CallInfo) -> tuple[int, SplitEvent] | None:
    """(line, later split) when ``c`` runs on full data before a hold-out split.

    Inside a helper function the parameter's role is unknown; use the line where the helper is
    called on full data instead.
    """
    role = a.role_of(c.receiver_expr, c.line)
    line = c.line
    if role is not Role.FULL:
        if c.func_scope is None or role is not Role.UNKNOWN:
            return None
        site = _full_data_call_site(a, c.func_scope)
        if site is None:
            return None
        line = site
    s = later_holdout_split(a, line)
    return (line, s) if s else None


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
        targets = resolved_target_columns(a)
        for c in a.calls:
            if c.func not in _AGG:
                continue
            key = groupby_target_key(a, c.receiver_expr, targets)
            if not key:
                continue
            if isinstance(c.stmt, ast.Expr):
                continue  # print(df.groupby(k)[t].mean()) is EDA, not a feature
            ctx = full_data_context(a, c)
            if ctx is None:
                continue
            _, s = ctx
            where = f" (in {c.func_scope}())" if c.func_scope else ""
            yield self.report(
                a, c.node,
                f"Mean-of-target encoding: groupby(...)[{key!r}].{c.func}(){where} over the full data before the "
                f"split on line {s.line}; test labels are baked into a feature.",
                confidence=HIGH if targets[key] >= 1.0 else MEDIUM,
            )
