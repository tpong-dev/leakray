"""LK010: rolling / lag features on the target without shift()."""

from __future__ import annotations

import ast
from collections.abc import Iterator

from ..dataflow import Analysis
from ..models import HIGH, MEDIUM, Finding, Severity
from ._util import ancestors, chain_has_call, resolved_target_columns
from .base import Rule, register

_WINDOW = {"rolling", "expanding", "ewm", "diff", "pct_change", "cumsum", "cummax", "cummin"}


def _target_key(node: ast.AST | None, targets: dict[str, float], target_vars: set[str]) -> str | None:
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
        return node.slice.value if node.slice.value in targets else None
    if isinstance(node, ast.Attribute) and node.attr in targets:
        return node.attr
    if isinstance(node, ast.Name) and node.id in target_vars:
        return node.id
    return None


@register
class UnshiftedTargetLag(Rule):
    code = "LK010"
    name = "unshifted-target-lag-feature"
    summary = "Rolling / lag / diff feature built from the target without shift()"
    severity = Severity.ERROR
    confidence = HIGH
    fix = 'Shift before the window so each row only sees the past: df["y"].shift(1).rolling(7).mean().'

    def check(self, a: Analysis) -> Iterator[Finding]:
        targets = resolved_target_columns(a)
        for c in a.calls_named(*_WINDOW):
            key = _target_key(c.receiver_expr, targets, a.target_vars)
            if key is None or chain_has_call(c.receiver_expr, "shift"):
                continue
            ups = ancestors(a, c.node)
            if any(isinstance(p, ast.Call) and isinstance(p.func, ast.Attribute) and p.func.attr == "shift"
                   for p in ups):
                continue
            stmt = ups[-1] if ups else None
            # building the target itself (y = df.price.pct_change()) is not a feature
            if isinstance(stmt, ast.Assign):
                tnames = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
                tkeys = [t.slice.value for t in stmt.targets if isinstance(t, ast.Subscript)
                         and isinstance(t.slice, ast.Constant)]
                if any(n in a.target_vars or n.lower() in {"y", "target"} for n in tnames) or key in tkeys:
                    continue
            known = targets.get(key, 1.0 if key in a.target_vars else 0.0) >= 1.0
            yield self.report(
                a, c.node,
                f".{c.func}() on the target {key!r} without .shift(): each row's feature includes its own "
                "label (and for windows, values from the same period).",
                confidence=HIGH if known else MEDIUM,
            )
