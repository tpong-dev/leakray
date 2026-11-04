"""Shared helpers for rules."""

from __future__ import annotations

import ast
import re

from ..dataflow import Analysis, CallInfo, Role, SplitEvent

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


# --- target resolution: a known target beats name hints ---------------------

_FIT_METHODS = {"fit", "fit_transform", "fit_predict", "partial_fit", "fit_resample"}
_CV_EVAL_Y_POS = {"cross_val_score": 2, "cross_validate": 2, "cross_val_predict": 2}


def _str_key(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
        return node.slice.value
    return None


def known_target_columns(a: Analysis) -> dict[str, float]:
    """Columns the file itself shows to be the label (all 1.0).

    ``y = df["c"]`` (dataflow) plus ``train_test_split(X, df["c"])``, ``stratify=df["c"]``,
    ``model.fit(X, df["c"])`` and ``cross_val_score(m, X, df["c"])``.
    """
    out: dict[str, float] = {c: 1.0 for c in a.target_columns}
    for c in a.calls:
        cands: list[ast.AST | None] = []
        if c.func == "train_test_split" and len(c.args) >= 2:
            cands.append(c.args[1])
        if c.func in _FIT_METHODS and len(c.args) >= 2:
            cands.append(c.args[1])
        pos = _CV_EVAL_Y_POS.get(c.func)
        if pos is not None and len(c.args) > pos:
            cands.append(c.args[pos])
        cands.append(c.kwargs.get("stratify"))
        cands.append(c.kwargs.get("y"))
        for node in cands:
            key = _str_key(node)
            if key:
                out.setdefault(key, 1.0)
    return out


def resolved_target_columns(a: Analysis) -> dict[str, float]:
    """Known targets when the file names one (no hints); otherwise name hints at 0.5."""
    known = known_target_columns(a)
    if known:
        return known
    return {s: 0.5 for s in a.string_constants if s.lower() in TARGET_NAME_HINTS}


# --- shuffled splitter objects & explicit date columns (used by LK002 / LK012) ---

_EXPLICIT_DATE_NAMES = {"date", "datetime", "timestamp", "ds", "time"}
_EXPLICIT_DATE_RE = re.compile(r"^(?:[a-z0-9]+_)+(?:date|datetime|timestamp|at)$|^date_[a-z0-9_]+$", re.I)
# splitters that ignore order (shuffle only matters for KFold-style ones)
SHUFFLE_KW_SPLITTERS = {"KFold", "StratifiedKFold"}
ALWAYS_RANDOM_SPLITTERS = {"ShuffleSplit", "StratifiedShuffleSplit", "RepeatedKFold", "RepeatedStratifiedKFold"}
CV_CONSUMERS = {
    "cross_val_score", "cross_validate", "cross_val_predict", "learning_curve", "validation_curve",
    "permutation_test_score", "GridSearchCV", "RandomizedSearchCV", "HalvingGridSearchCV",
    "HalvingRandomSearchCV", "BayesSearchCV", "TuneSearchCV",
}


def explicit_date_column(name: str) -> bool:
    """Strict: ``date``/``datetime``/``timestamp``/``ds``/``time`` or ``*_date``, ``date_*``, ``*_at`` ..."""
    return name.isidentifier() and (name.lower() in _EXPLICIT_DATE_NAMES or bool(_EXPLICIT_DATE_RE.match(name)))


def column_keys(a: Analysis) -> set[str]:
    """String literals used as column keys: ``df["c"]``, ``df[["a", "c"]]``, ``.loc[:, "c"]``."""
    out: set[str] = set()
    for n in ast.walk(a.tree):
        if not isinstance(n, ast.Subscript):
            continue
        sl = n.slice
        parts = sl.elts if isinstance(sl, (ast.Tuple, ast.List)) else [sl]
        for p in parts:
            elts = p.elts if isinstance(p, ast.List) else [p]
            for e in elts:
                if isinstance(e, ast.Constant) and isinstance(e.value, str):
                    out.add(e.value)
    return out


def splitter_constructions(
    a: Analysis, classes: set[str], need_shuffle: set[str] | frozenset[str] = frozenset()
) -> list[CallInfo]:
    """Calls constructing a CV splitter of ``classes``; those in ``need_shuffle`` only with shuffle=True."""
    out = []
    for c in a.calls:
        if c.func not in classes:
            continue
        if c.func in need_shuffle and c.kwarg_const("shuffle") is not True:
            continue
        out.append(c)
    return out


def assigned_name(call_node: ast.AST, a: Analysis) -> str | None:
    """``kf = KFold(...)`` -> "kf" (only when the call is the whole right-hand side)."""
    stmt = a.stmt_of(call_node)
    if isinstance(stmt, ast.Assign) and stmt.value is call_node and len(stmt.targets) == 1:
        t = stmt.targets[0]
        if isinstance(t, ast.Name):
            return t.id
    return None
