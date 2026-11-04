"""LK003: target column left inside the feature matrix."""

from __future__ import annotations

import ast
from collections.abc import Iterator

from ..dataflow import Analysis
from ..models import HIGH, MEDIUM, Finding, Severity
from .base import Rule, register

# Calls that return (roughly) the same columns as their receiver.
_PASSTHROUGH = {"copy", "dropna", "fillna", "drop", "reset_index", "sort_values", "astype", "values",
                "to_numpy", "drop_duplicates", "sort_index", "set_axis", "infer_objects"}
# Calls on a column list / Index that keep its elements.
_LIST_PASSTHROUGH = {"tolist", "to_list", "copy", "drop", "difference", "unique", "to_numpy"}
_NUMERIC_DTYPES = {"number", "float", "int", "float64", "int64", "float32", "int32", "numeric"}


def _latest_assignment(a: Analysis, name: str, line: int) -> ast.expr | None:
    best = None
    for asg in a.assignments:
        if asg.line < line and name in asg.targets and len(asg.targets) == 1:
            best = asg.value
    return best


def _latest_assignment_line(a: Analysis, name: str, line: int) -> int:
    best = 0
    for asg in a.assignments:
        if asg.line < line and name in asg.targets:
            best = asg.line
    return best


def _const_str(a: Analysis, node: ast.expr, line: int, depth: int = 0) -> str | None:
    """A string literal, or a Name bound to one (``TARGET = "price"``)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name) and depth < 3:
        val = _latest_assignment(a, node.id, line)
        if val is not None:
            return _const_str(a, val, line, depth + 1)
    return None


def _mentions(a: Analysis, node: ast.AST, line: int, depth: int = 0) -> set[str]:
    """String literals under ``node``, resolving Names bound to strings or string lists."""
    out: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            out.add(n.value)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and depth < 3:
            val = _latest_assignment(a, n.id, line)
            if isinstance(val, ast.Constant) and isinstance(val.value, str):
                out.add(val.value)
            elif isinstance(val, (ast.List, ast.Tuple, ast.Set, ast.Name)):
                out |= _mentions(a, val, line, depth + 1)
    return out


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
        key = _const_str(a, expr.slice, line)
        if key is not None:
            return expr.value.id, key
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


def _is_columns_of_frame(node: ast.expr) -> bool:
    """``df.columns`` / ``list(df)``: every column of a frame."""
    if isinstance(node, ast.Attribute) and node.attr == "columns":
        return True
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "list"
            and len(node.args) == 1 and isinstance(node.args[0], ast.Name)):
        return True
    return False


def _list_has_key(a: Analysis, node: ast.expr, key: str, line: int, depth: int = 0) -> bool | None:
    """Does a column-list expression contain ``key``? None when unknown."""
    if depth > 6:
        return None
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        if any(isinstance(e, ast.Starred) for e in node.elts):
            return True if key in _mentions(a, node, line) else None
        return any(_const_str(a, e, line) == key for e in node.elts)
    if isinstance(node, ast.Constant):
        return None  # single column: a Series, not a frame of features
    if isinstance(node, ast.Name):
        val = _latest_assignment(a, node.id, line)
        if val is None:
            return None
        res = _list_has_key(a, val, key, line, depth + 1)
        if res:
            # features.remove(TARGET) / features.pop(...) after the assignment removes it
            since = _latest_assignment_line(a, node.id, line)
            for c in a.calls_named("remove", "pop", "discard"):
                if c.receiver == node.id and since <= c.line < line:
                    if c.func == "pop" or key in _mentions(a, c.node, line):
                        return False
        return res
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _list_has_key(a, node.left, key, line, depth + 1)
        right = _list_has_key(a, node.right, key, line, depth + 1)
        if left or right:
            return True
        return None if left is None or right is None else False
    if isinstance(node, (ast.ListComp, ast.GeneratorExp, ast.SetComp)):
        if key in _mentions(a, node, line):
            return False  # [c for c in df.columns if c != TARGET]
        gen = node.generators[0]
        if len(node.generators) != 1 or not _is_columns_of_frame(gen.iter):
            return None
        if not (isinstance(node.elt, ast.Name) and isinstance(gen.target, ast.Name)
                and node.elt.id == gen.target.id):
            return None
        # only exclusion filters (!=, not in) that do not name the target keep it
        for cond in gen.ifs:
            if not (isinstance(cond, ast.Compare)
                    and all(isinstance(op, (ast.NotEq, ast.NotIn, ast.IsNot)) for op in cond.ops)):
                return None
        return True
    if _is_columns_of_frame(node):
        return True
    if isinstance(node, ast.Call):
        if key in _mentions(a, node, line):
            return False  # df.columns.drop(TARGET), .difference([TARGET])
        if isinstance(node.func, ast.Name) and node.func.id in {"list", "sorted", "tuple"} and len(node.args) == 1:
            return _list_has_key(a, node.args[0], key, line, depth + 1)
        if isinstance(node.func, ast.Attribute) and node.func.attr in _LIST_PASSTHROUGH:
            if node.func.attr in {"drop", "difference"} and not (node.args or node.keywords):
                return None
            return _list_has_key(a, node.func.value, key, line, depth + 1)
        return None
    return None


def _is_full_slice(node: ast.expr) -> bool:
    return isinstance(node, ast.Slice) and node.lower is None and node.upper is None and node.step is None


def _select_dtypes_keeps_numeric(call: ast.Call) -> bool:
    def names(v: ast.expr) -> set[str]:
        out: set[str] = set()
        for n in ast.walk(v):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                out.add(n.value.lower())
            elif isinstance(n, ast.Attribute):
                out.add(n.attr.lower())
            elif isinstance(n, ast.Name):
                out.add(n.id.lower())
        return out

    include = call.args[0] if call.args else None
    exclude = call.args[1] if len(call.args) > 1 else None
    for kw in call.keywords:
        if kw.arg == "include":
            include = kw.value
        elif kw.arg == "exclude":
            exclude = kw.value
    if include is not None:
        return bool(names(include) & _NUMERIC_DTYPES)
    if exclude is not None:
        return not (names(exclude) & _NUMERIC_DTYPES)
    return False


def _x_has_target(a: Analysis, expr: ast.expr, frame: str, key: str, line: int, depth: int = 0) -> float | None:
    """Confidence that the features ``expr`` still hold ``frame[key]``; None when they don't (or unknown)."""
    if depth > 6:
        return None
    if isinstance(expr, ast.Name):
        if expr.id == frame:
            # popped / dropped in place earlier?
            for c in a.calls_named("pop", "drop"):
                if c.receiver == frame and c.line < line and key in _mentions(a, c.node, line) and (
                    c.func == "pop" or c.kwarg_const("inplace") is True
                ):
                    return None
            return HIGH
        src = _split_input(a, expr.id, line)
        if src:
            return _x_has_target(a, ast.Name(id=src, ctx=ast.Load()), frame, key, line, depth + 1)
        val = _latest_assignment(a, expr.id, line)
        if val is None:
            return None
        return _x_has_target(a, val, frame, key, line, depth + 1)
    if isinstance(expr, ast.Subscript):
        base = expr.value
        if isinstance(base, ast.Attribute) and base.attr in {"iloc", "loc"}:
            sl = expr.slice
            parts = list(sl.elts) if isinstance(sl, ast.Tuple) else [sl]
            if not _is_full_slice(parts[0]):
                return None  # row subsets are tracked as splits elsewhere
            if len(parts) == 1 or (len(parts) == 2 and _is_full_slice(parts[1])):
                return _x_has_target(a, base.value, frame, key, line, depth + 1)
            if base.attr == "loc" and len(parts) == 2 and _list_has_key(a, parts[1], key, line):
                return HIGH
            return None
        return HIGH if _list_has_key(a, expr.slice, key, line) else None
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
        attr = expr.func.attr
        if attr in {"drop", "pop"} and key in _mentions(a, expr, line):
            return None  # df.drop(columns=[TARGET])
        if attr in _PASSTHROUGH:
            return _x_has_target(a, expr.func.value, frame, key, line, depth + 1)
        if attr == "select_dtypes" and _select_dtypes_keeps_numeric(expr):
            inner = _x_has_target(a, expr.func.value, frame, key, line, depth + 1)
            return None if inner is None else min(inner, MEDIUM)
        return None
    if isinstance(expr, ast.Attribute) and expr.attr == "values":
        return _x_has_target(a, expr.value, frame, key, line, depth + 1)
    return None


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
            conf = _x_has_target(a, fit.data_args[0], frame, key, fit.line)
            if conf is not None:
                extra = " (if the target is numeric, select_dtypes keeps it)" if conf < HIGH else ""
                yield self.report(
                    a, fit.call.node,
                    f"The target column {key!r} (y = {frame}[{key!r}]) is still inside the features passed "
                    f"to {fit.estimator or 'the model'}.{fit.method}(); the model can read the answer{extra}.",
                    confidence=conf,
                    column=key,
                )
