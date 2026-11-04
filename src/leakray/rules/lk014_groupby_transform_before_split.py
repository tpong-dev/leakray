"""LK014: aggregate features over the full data before the split."""

from __future__ import annotations

import ast
from collections.abc import Iterator

from ..dataflow import Analysis, CallInfo, Role
from ..models import LOW, MEDIUM, Finding, Severity
from ._util import ancestors, later_holdout_split, resolved_target_columns
from .base import Rule, register
from .lk008_target_encoding_full_data import groupby_target_key

# past-only / row-local group operations: no statistic over other rows' future values
_SAFE_TRANSFORMS = {"shift", "cumsum", "cumcount", "cummax", "cummin", "cumprod", "diff", "ffill", "bfill",
                    "first", "head", "nth"}
# group aggregations whose result is a per-group table that can be merged / mapped back
_GROUP_AGGS = {"agg", "aggregate", "mean", "median", "std", "var", "sum", "count", "nunique", "min", "max",
               "size", "quantile", "sem", "skew"}
# column statistics used for whole-column normalisation: df[c] / df[c].max()
_COLUMN_STATS = {"max", "min", "mean", "std", "sum", "median", "var", "quantile"}
_MAP_BACK = {"map", "merge", "join", "replace"}


def _is_call(n: ast.AST, attr: str) -> bool:
    return isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == attr


def _has_groupby(expr: ast.AST | None) -> bool:
    return expr is not None and any(_is_call(n, "groupby") for n in ast.walk(expr))


def _assigned_name(a: Analysis, c: CallInfo) -> str | None:
    """``freq = <expr containing c>`` -> "freq"."""
    stmt = c.stmt
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
        return stmt.targets[0].id
    return None


def _assigned_into_frame(c: CallInfo) -> bool:
    """``df["c"] = ...`` / ``df.loc[:, "c"] = ...`` / ``df = df.assign(...)``."""
    stmt = c.stmt
    if isinstance(stmt, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
        tgts = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
        if any(isinstance(t, ast.Subscript) for t in tgts):
            return True
        return isinstance(stmt.value, ast.Call) and any(_is_call(n, "assign") for n in ast.walk(stmt.value))
    return False


def _mapped_back_inline(a: Analysis, c: CallInfo) -> bool:
    for p in ancestors(a, c.node):
        if isinstance(p, ast.Call) and isinstance(p.func, ast.Attribute) and p.func.attr in _MAP_BACK:
            return True
        if isinstance(p, ast.Call) and isinstance(p.func, (ast.Attribute, ast.Name)) and \
                getattr(p.func, "attr", getattr(p.func, "id", "")) == "merge":
            return True
    return False


def _mapped_back_later(a: Analysis, name: str, line: int) -> int | None:
    """Line of a later ``x.map(name)`` / ``x.merge(name)`` / ``pd.merge(x, name)`` / ``x.join(name)``."""
    for c in a.calls:
        if c.line <= line or c.func not in _MAP_BACK:
            continue
        parts = list(c.args) + list(c.kwargs.values())
        if c.func in {"merge", "join"} and c.receiver_expr is not None:
            parts.append(c.receiver_expr)
        if any(isinstance(n, ast.Name) and n.id == name for p in parts for n in ast.walk(p)):
            return c.line
    return None


def _used_as_feature(a: Analysis, c: CallInfo) -> bool:
    if _mapped_back_inline(a, c) or _assigned_into_frame(c):
        return True
    name = _assigned_name(a, c)
    return name is not None and _mapped_back_later(a, name, c.line) is not None


def _subscript_col(n: ast.AST) -> tuple[str, str] | None:
    if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and isinstance(n.slice, ast.Constant) \
            and isinstance(n.slice.value, str):
        return n.value.id, n.slice.value
    return None


def _normalisation(a: Analysis, stmt: ast.stmt) -> ast.Call | None:
    """``df["c"] = df["c"] / df["c"].max()`` or ``(df[c] - df[c].mean()) / df[c].std()``."""
    if not (isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Subscript) for t in stmt.targets)):
        return None
    for b in ast.walk(stmt.value):
        if not (isinstance(b, ast.BinOp) and isinstance(b.op, (ast.Div, ast.Sub))):
            continue
        for side, other in ((b.right, b.left), (b.left, b.right)):
            if not (isinstance(side, ast.Call) and isinstance(side.func, ast.Attribute)
                    and side.func.attr in _COLUMN_STATS and not side.args):
                continue
            col = _subscript_col(side.func.value)
            if col and any(_subscript_col(n) == col for n in ast.walk(other)):
                return side
    return None


@register
class GroupbyTransformBeforeSplit(Rule):
    code = "LK014"
    name = "groupby-transform-before-split"
    summary = "groupby().transform / frequency encoding computed over the full data before the split"
    severity = Severity.WARNING
    confidence = MEDIUM
    fix = (
        "Compute group statistics and frequency counts on the training rows only, then merge/map them "
        "onto the test rows (unseen groups get a default)."
    )

    def check(self, a: Analysis) -> Iterator[Finding]:
        targets = resolved_target_columns(a)
        seen_norm: set[int] = set()
        for c in a.calls:
            if isinstance(c.stmt, ast.Expr):
                continue  # print(...) / bare display: EDA, nothing assigned back
            conf = MEDIUM
            if c.func == "transform" and _has_groupby(c.receiver_expr):
                if groupby_target_key(a, c.receiver_expr, targets):
                    continue  # target means are LK008
                arg = c.args[0] if c.args else None
                if isinstance(arg, ast.Constant) and arg.value in _SAFE_TRANSFORMS:
                    continue
                kind = "groupby().transform()"
            elif c.func == "rank" and _has_groupby(c.receiver_expr):
                if not _used_as_feature(a, c):
                    continue
                kind = "groupby().rank()"
            elif c.func in _GROUP_AGGS and _has_groupby(c.receiver_expr):
                if groupby_target_key(a, c.receiver_expr, targets):
                    continue  # LK008
                if not (_mapped_back_inline(a, c) or (
                        (name := _assigned_name(a, c)) and _mapped_back_later(a, name, c.line))):
                    continue
                kind = f"groupby().{c.func}() merged/mapped back"
            elif c.func == "value_counts":
                if not _used_as_feature(a, c):
                    continue
                kind = "frequency encoding with value_counts()"
                conf = LOW
            elif c.func in _COLUMN_STATS and id(c.stmt) not in seen_norm and _normalisation(a, c.stmt) is c.node:
                seen_norm.add(id(c.stmt))
                kind = f"column normalisation by .{c.func}()"
                conf = LOW
            else:
                continue
            if a.role_of(c.receiver_expr, c.line) is not Role.FULL:
                continue
            s = later_holdout_split(a, c.line)
            if s is None:
                continue
            yield self.report(
                a, c.node,
                f"{kind} over the full data before the split on line {s.line}: group statistics include "
                "test rows.",
                confidence=conf,
            )
