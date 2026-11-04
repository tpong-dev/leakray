"""LK014: aggregate features over the full data before the split."""

from __future__ import annotations

import ast
from collections.abc import Iterator

from ..dataflow import Analysis, Role
from ..models import LOW, MEDIUM, Finding, Severity
from ._util import ancestors, later_holdout_split, target_columns
from .base import Rule, register
from .lk008_target_encoding_full_data import groupby_target_key

_SAFE_TRANSFORMS = {"shift", "cumsum", "cumcount", "diff", "ffill", "bfill", "rank"}


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
        targets = target_columns(a)
        for c in a.calls:
            if c.func == "transform" and c.receiver_expr is not None:
                has_gb = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                             and n.func.attr == "groupby" for n in ast.walk(c.receiver_expr))
                if not has_gb or groupby_target_key(a, c.receiver_expr, targets):
                    continue  # target means are LK008
                arg = c.args[0] if c.args else None
                if isinstance(arg, ast.Constant) and arg.value in _SAFE_TRANSFORMS:
                    continue
                kind = "groupby().transform()"
            elif c.func == "value_counts":
                if not any(isinstance(p, ast.Call) and isinstance(p.func, ast.Attribute) and p.func.attr == "map"
                           for p in ancestors(a, c.node)):
                    continue
                kind = "frequency encoding with value_counts()"
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
                confidence=MEDIUM if kind.startswith("groupby") else LOW,
            )
