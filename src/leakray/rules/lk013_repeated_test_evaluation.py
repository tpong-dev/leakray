"""LK013: the test set is scored repeatedly to pick a model."""

from __future__ import annotations

import ast
from collections.abc import Iterator

from ..dataflow import Analysis, Role
from ..models import LOW, MEDIUM, Finding, Severity
from ._util import cv_loop_ranges
from .base import Rule, register

_EVAL = {"score", "predict", "predict_proba", "evaluate", "decision_function"}


def _is_metric(func: str) -> bool:
    return func.endswith(("_score", "_error", "_loss")) or func in {"log_loss", "classification_report"}


@register
class RepeatedTestEvaluation(Rule):
    code = "LK013"
    name = "repeated-test-evaluation"
    summary = "Test set evaluated inside a loop over models / hyperparameters to pick the best one"
    severity = Severity.WARNING
    confidence = MEDIUM
    fix = (
        "Compare models on a validation split or with cross-validation on the training data; "
        "evaluate the single chosen model on the test set once."
    )

    def check(self, a: Analysis) -> Iterator[Finding]:
        loops = cv_loop_ranges(a)
        selects = any(isinstance(n, ast.Name) and "best" in n.id.lower() for n in ast.walk(a.tree)) or any(
            c.func in {"max", "argmax", "idxmax", "sort_values", "sorted", "nlargest"} for c in a.calls)
        if not selects:
            return
        test_evals = []
        for c in a.calls:
            if not (c.func in _EVAL or _is_metric(c.func)):
                continue
            args = list(c.args) + list(c.kwargs.values())
            if any(a.role_of(x, c.line) is Role.TEST for x in args):
                if any(lo <= c.line <= hi for lo, hi in loops):
                    continue  # per-fold scoring in a CV loop is fine
                test_evals.append(c)
        in_loop = [c for c in test_evals if c.loop_depth > 0]
        if in_loop:
            c = in_loop[0]
            yield self.report(
                a, c.node,
                f"Test-set {c.func}() inside a loop, and the best result is selected: the test score becomes "
                "a tuning target and over-states real performance.",
            )
            return
        receivers = {c.receiver for c in test_evals if c.func in _EVAL and c.receiver}
        if len(receivers) >= 3:
            c = test_evals[0]
            yield self.report(a, c.node, f"{len(receivers)} models are scored on the test set and the best is "
                              "picked; use a validation set for model choice.", confidence=LOW)
