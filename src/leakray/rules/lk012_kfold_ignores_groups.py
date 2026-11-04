"""LK012: K-fold / random split ignoring a group or entity id column."""

from __future__ import annotations

import re
from collections.abc import Iterator

from ..dataflow import Analysis
from ..models import LOW, MEDIUM, Finding, Severity
from ._util import assigned_name, splitter_constructions
from .base import Rule, register

_STRONG = re.compile(
    r"^(patient|user|customer|subject|session|household|player|person|participant|client|member|driver|"
    r"student|account|device|visitor)(_?id|_?no)?$", re.I)
_WEAK = re.compile(r"^[a-z]+_id$", re.I)
_UNGROUPED = {"train_test_split", "KFold", "StratifiedKFold", "ShuffleSplit", "StratifiedShuffleSplit",
              "RepeatedKFold", "RepeatedStratifiedKFold"}


@register
class KFoldIgnoresGroups(Rule):
    code = "LK012"
    name = "kfold-ignores-groups"
    summary = "Rows are split at random although a group/id column means the same entity appears many times"
    severity = Severity.WARNING
    confidence = MEDIUM
    fix = (
        "Split by entity so one patient/user never lands on both sides: GroupKFold / StratifiedGroupKFold / "
        "GroupShuffleSplit with groups=df[\"<id>\"]."
    )

    def check(self, a: Analysis) -> Iterator[Finding]:
        if any(c.func.startswith(("Group", "StratifiedGroup", "LeaveOneGroup", "LeavePGroups")) for c in a.calls):
            return
        if any("groups" in c.kwargs for c in a.calls):
            return
        strong = sorted(s for s in a.string_constants if _STRONG.match(s))
        weak = sorted(s for s in a.string_constants if _WEAK.match(s) and s not in strong)
        if not strong and not weak:
            return
        col = (strong or weak)[0]
        conf = MEDIUM if strong else LOW
        entity = col.split("_")[0]
        reported_receivers: set[str] = set()
        seen: set[int] = set()
        for s in a.splits:
            name = "train_test_split" if s.kind == "train_test_split" else s.splitter
            if s.kind not in {"train_test_split", "cv_split"} or name not in _UNGROUPED:
                continue
            if s.kind == "cv_split" and s.call and s.call.receiver:
                reported_receivers.add(s.call.receiver)
            seen.add(s.line)
            yield self.report(
                a, s.node,
                f"{name} splits rows at random, but column {col!r} suggests repeated entities: the same "
                f"{entity} can appear in train and test, so the model is graded on people it has seen.",
                confidence=conf,
            )
        # splitter objects used without a .split() loop, e.g. cross_val_score(..., cv=KFold(5, shuffle=True))
        for c in splitter_constructions(a, _UNGROUPED - {"train_test_split"}):
            if assigned_name(c.node, a) in reported_receivers or c.line in seen:
                continue
            seen.add(c.line)
            yield self.report(
                a, c.node,
                f"{c.func} assigns rows to folds without groups=, but column {col!r} suggests repeated "
                f"entities: the same {entity} can land in both the training and validation folds.",
                confidence=conf,
            )
