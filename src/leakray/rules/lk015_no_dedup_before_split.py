"""LK015: no duplicate removal before a random split."""

from __future__ import annotations

from collections.abc import Iterator

from ..dataflow import Analysis
from ..models import LOW, Finding, Severity
from .base import Rule, register


@register
class NoDedupBeforeRandomSplit(Rule):
    code = "LK015"
    name = "no-dedup-before-random-split"
    summary = "Random split without removing duplicate rows first (duplicates can sit in train and test)"
    severity = Severity.INFO
    confidence = LOW
    fix = "Run df = df.drop_duplicates() (and check near-duplicates) before splitting, or split by group."

    def check(self, a: Analysis) -> Iterator[Finding]:
        if any(c.func in {"drop_duplicates", "duplicated", "unique", "distinct", "dropDuplicates"} for c in a.calls):
            return
        for s in a.splits:
            if s.kind == "train_test_split" and s.is_random:
                yield self.report(
                    a, s.node,
                    "Random split with no drop_duplicates() in this file: identical rows can land in both "
                    "train and test and inflate the score.",
                )
                return
