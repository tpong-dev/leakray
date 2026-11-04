"""LK002: random split on time-ordered data."""

from __future__ import annotations

import ast
from collections.abc import Iterator

from ..dataflow import Analysis
from ..models import HIGH, MEDIUM, Finding, Severity
from ._util import (
    ALWAYS_RANDOM_SPLITTERS,
    CV_CONSUMERS,
    SHUFFLE_KW_SPLITTERS,
    assigned_name,
    column_keys,
    explicit_date_column,
    splitter_constructions,
)
from .base import Rule, register


def _datetime_evidence(a: Analysis) -> list[tuple[int, str]]:
    """Explicit datetime evidence, minus ``sklearn.utils.resample`` (bootstrap, not time resampling)."""
    sk_resample = {c.line for c in a.calls if c.func == "resample" and c.name.startswith("sklearn")}
    return [(ln, why) for ln, why in a.datetime_evidence if not (why == "resample(...)" and ln in sk_resample)]


@register
class RandomSplitOnTimeSeries(Rule):
    code = "LK002"
    name = "random-split-on-time-series"
    summary = "Random train/test split or shuffled K-fold on data that has a datetime column or index"
    severity = Severity.WARNING
    confidence = HIGH
    fix = (
        "Split by time: sort by the date and cut at a cutoff date (train = past, test = future), "
        "use TimeSeriesSplit for CV, or pass shuffle=False to train_test_split on time-sorted data."
    )

    def check(self, a: Analysis) -> Iterator[Finding]:
        if any(c.func == "TimeSeriesSplit" for c in a.calls):
            return
        dt_ev = _datetime_evidence(a)
        strong = bool(dt_ev)
        date_cols = sorted(k for k in column_keys(a) if explicit_date_column(k))
        if not strong and not date_cols:
            return
        evidence = f"datetime evidence on line {dt_ev[0][0]} ({dt_ev[0][1]})" if strong \
            else f"date column {date_cols[0]!r}"
        conf = HIGH if strong else MEDIUM
        reported_receivers: set[str] = set()
        seen: set[int] = set()

        for s in a.splits:
            if s.kind not in {"train_test_split", "cv_split", "random_sample"} or not s.is_random:
                continue
            if s.kind == "train_test_split" and s.call and s.call.kwarg_const("shuffle") is False:
                continue
            if s.kind == "cv_split" and s.call and s.call.receiver:
                reported_receivers.add(s.call.receiver)
            what = {"train_test_split": "train_test_split()", "cv_split": f"{s.splitter or 'K-fold'} split",
                    "random_sample": ".sample()"}[s.kind]
            seen.add(s.line)
            yield self.report(
                a, s.node,
                f"Random {what} on time-ordered data ({evidence}): the model trains on the future "
                "and is tested on the past, so scores look better than in production.",
                confidence=conf,
            )

        # shuffled splitter objects, e.g. cross_val_score(..., cv=KFold(5, shuffle=True))
        for c in splitter_constructions(a, SHUFFLE_KW_SPLITTERS | ALWAYS_RANDOM_SPLITTERS, SHUFFLE_KW_SPLITTERS):
            if assigned_name(c.node, a) in reported_receivers or c.line in seen:
                continue
            seen.add(c.line)
            yield self.report(
                a, c.node,
                f"Shuffled {c.func} on time-ordered data ({evidence}): folds mix past and future rows, "
                "so the model is validated on the past after training on the future.",
                confidence=conf,
            )

        # cross_val_score / GridSearchCV with cv=<int> or no cv: plain K-fold that ignores time order
        for c in a.calls:
            if c.func not in CV_CONSUMERS or c.line in seen:
                continue
            cv = c.kwargs.get("cv")
            if cv is not None and not (isinstance(cv, ast.Constant) and isinstance(cv.value, int)):
                continue
            seen.add(c.line)
            how = f"cv={cv.value!r}" if isinstance(cv, ast.Constant) else "the default cv"
            yield self.report(
                a, c.node,
                f"{c.func}() with {how} uses plain K-fold on time-ordered data ({evidence}): most folds "
                "train on rows that come after the test rows. Use cv=TimeSeriesSplit(...).",
                confidence=MEDIUM,
            )
