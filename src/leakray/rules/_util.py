"""Shared helpers for rules."""

from __future__ import annotations

import ast
import re

from ..dataflow import Analysis, Role, SplitEvent

# Column names that very often hold the label, used when no ``y = df[...]`` is seen.
TARGET_NAME_HINTS = {
    "target", "label", "labels", "y", "class", "outcome", "price", "saleprice", "churn",
    "survived", "fraud", "is_fraud", "default", "sales", "revenue", "demand",
}

_DATE_TOKENS = {"date", "time", "timestamp", "datetime", "month", "year", "day", "week", "ts", "dt", "period"}


def date_like_column(name: str) -> bool:
    tokens = [t for t in re.split(r"[_\W\d]+", name.lower()) if t]
    return bool(tokens) and any(t in _DATE_TOKENS for t in tokens) and name.isidentifier()


def target_columns(a: Analysis, include_hints: bool = True) -> dict[str, float]:
    """Column name -> how sure we are that it is the target (1.0 known, 0.5 hint)."""
    out: dict[str, float] = {c: 1.0 for c in a.target_columns}
    if include_hints:
        for s in a.string_constants:
            if s.lower() in TARGET_NAME_HINTS and s not in out:
                out[s] = 0.5
    return out


def subscript_key(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
        return node.slice.value
    return None


def chain_has_call(node: ast.AST | None, attr: str) -> bool:
    """Does the method chain inside ``node`` contain ``.attr(...)``?"""
    if node is None:
        return False
    return any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == attr for n in ast.walk(node)
    )


def ancestors(a: Analysis, node: ast.AST, stop_at_stmt: bool = True) -> list[ast.AST]:
    out = []
    cur = a.parent(node)
    while cur is not None:
        out.append(cur)
        if stop_at_stmt and isinstance(cur, ast.stmt):
            break
        cur = a.parent(cur)
    return out


def later_holdout_split(a: Analysis, line: int) -> SplitEvent | None:
    """First non-CV split after ``line``."""
    later = [s for s in a.splits if s.line > line and s.kind != "cv_split"]
    return min(later, key=lambda s: s.line) if later else None


def later_cv(a: Analysis, line: int) -> int | None:
    """Line of the first CV evaluation (cross_val_score, SearchCV.fit, CV loop) after ``line``."""
    lines = [c.line for c in a.cv_evals if c.line > line]
    lines += [s.line for s in a.splits if s.kind == "cv_split" and s.line > line]
    return min(lines) if lines else None


def concat_train_test(a: Analysis) -> dict[str, int]:
    """Variables built by concatenating train and test data: name -> line."""
    out: dict[str, int] = {}
    for asg in a.assignments:
        if len(asg.targets) != 1:
            continue
        for n in ast.walk(asg.value):
            if is_concat_train_test(a, n, asg.line):
                out[asg.targets[0]] = asg.line
                break
    return out


def is_concat_train_test(a: Analysis, node: ast.AST, line: int) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
    if func not in {"concat", "vstack", "row_stack", "concatenate", "append", "union"}:
        return False
    parts: list[ast.AST] = list(node.args)
    if isinstance(node.func, ast.Attribute) and func == "append":
        parts.append(node.func.value)
    roles = set()
    for p in parts:
        for name in a.data_names_in(p):
            roles.add(a.role_at(name, line))
    return Role.TEST in roles and (Role.TRAIN in roles or Role.FULL in roles)


def cv_loop_ranges(a: Analysis) -> list[tuple[int, int]]:
    """Line ranges of ``for tr, te in kf.split(...)`` loops."""
    out = []
    for s in a.splits:
        if s.kind == "cv_split":
            stmt = a.stmt_of(s.node)
            if isinstance(stmt, (ast.For, ast.AsyncFor)):
                out.append((stmt.lineno, stmt.end_lineno or stmt.lineno))
    return out
