"""LK003: target column left inside the feature matrix."""

from __future__ import annotations

import ast
from collections.abc import Iterator

from ..dataflow import Analysis
from ..models import HIGH, Finding, Severity
from .base import Rule, register


def _latest_assignment(a: Analysis, name: str, line: int) -> ast.expr | None:
    best = None
    for asg in a.assignments:
        if asg.line < line and name in asg.targets and len(asg.targets) == 1:
            best = asg.value
    return best


def _split_input(a: Analysis, name: str, line: int) -> str | None:
    for s in a.splits:
        if s.line < line and s.kind == "train_test_split":
            for side in (s.train, s.test):
                if name in side:
                    idx = side.index(name)
                    if idx < len(s.inputs):
                        return s.inputs[idx]
    return None


def _y_key(a: Analysis, expr: ast.expr, line: int, depth: int = 0) -> tuple[str, str] | None:
    """(frame name, target column) for the y argument, if we can tell."""
    if isinstance(expr, ast.Subscript) and isinstance(expr.value, ast.Name):
        if isinstance(expr.slice, ast.Constant) and isinstance(expr.slice.value, str):
            return expr.value.id, expr.slice.value
    if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name) and expr.attr not in {"values"}:
        if expr.attr in a.target_columns:
            return expr.value.id, expr.attr
    if isinstance(expr, ast.Attribute) and expr.attr == "values":
        return _y_key(a, expr.value, line, depth + 1)
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute) and expr.func.attr == "pop":
        return None  # pop removes the column: safe
    if isinstance(expr, ast.Name) and depth < 4:
        src = _split_input(a, expr.id, line)
        if src:
            return _y_key(a, ast.Name(id=src, ctx=ast.Load()), line, depth + 1)
        val = _latest_assignment(a, expr.id, line)
        if val is not None:
            return _y_key(a, val, line, depth + 1)
    return None


def _x_has_target(a: Analysis, expr: ast.expr, frame: str, key: str, line: int, depth: int = 0) -> bool:
    if depth > 5:
        return False
    if isinstance(expr, ast.Name):
        if expr.id == frame:
            # popped earlier?
            for c in a.calls_named("pop", "drop"):
                if c.receiver == frame and c.line < line and key in _consts(c.node) and (
                    c.func == "pop" or c.kwarg_const("inplace") is True
                ):
                    return False
            return True
        src = _split_input(a, expr.id, line)
        if src:
            return _x_has_target(a, ast.Name(id=src, ctx=ast.Load()), frame, key, line, depth + 1)
        val = _latest_assignment(a, expr.id, line)
        if val is None:
            return False
        return _x_has_target(a, val, frame, key, line, depth + 1)
    if key in _consts(expr):
        # df[[..., "target"]] keeps it; anything else mentioning the key (drop, !=) removes it
        if isinstance(expr, ast.Subscript) and isinstance(expr.slice, (ast.List, ast.Tuple)):
            return True
        return False
    if isinstance(expr, ast.Subscript) and isinstance(expr.slice, (ast.List, ast.Tuple, ast.Constant)):
        return False  # explicit column list without the key
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
        if expr.func.attr in {"copy", "dropna", "fillna", "drop", "reset_index", "sort_values", "astype", "values",
                              "to_numpy", "drop_duplicates"}:
            return _x_has_target(a, expr.func.value, frame, key, line, depth + 1)
        return False
    if isinstance(expr, ast.Attribute) and expr.attr == "values":
        return _x_has_target(a, expr.value, frame, key, line, depth + 1)
    return False


def _consts(node: ast.AST) -> set[str]:
    return {n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)}


@register
class TargetInFeatures(Rule):
    code = "LK003"
    name = "target-in-features"
    summary = "Target column is still inside the feature matrix passed to fit"
    severity = Severity.ERROR
    confidence = HIGH
    fix = 'Drop the target from the features: X = df.drop(columns=["target"]) (or y = df.pop("target")).'

    def check(self, a: Analysis) -> Iterator[Finding]:
        for fit in a.fits:
            if len(fit.data_args) < 2 or fit.method not in {"fit", "fit_transform", "fit_predict"}:
                continue
            yk = _y_key(a, fit.data_args[1], fit.line)
            if yk is None:
                continue
            frame, key = yk
            if _x_has_target(a, fit.data_args[0], frame, key, fit.line):
                yield self.report(
                    a, fit.call.node,
                    f"The target column {key!r} (y = {frame}[{key!r}]) is still inside the features passed "
                    f"to {fit.estimator or 'the model'}.{fit.method}(); the model can read the answer.",
                    column=key,
                )
