"""LK002: random split on time-ordered data."""

from __future__ import annotations

from collections.abc import Iterator

from ..dataflow import Analysis
from ..models import HIGH, MEDIUM, Finding, Severity
from ._util import date_like_column
from .base import Rule, register


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
        strong = bool(a.datetime_evidence)
        date_cols = sorted({*a.datetime_columns, *(s for s in a.string_constants if date_like_column(s))})
        if not strong and not date_cols:
            return
        for s in a.splits:
            if s.kind not in {"train_test_split", "cv_split", "random_sample"} or not s.is_random:
                continue
            if s.kind == "train_test_split" and s.call and s.call.kwarg_const("shuffle") is False:
                continue
            what = {"train_test_split": "train_test_split()", "cv_split": f"{s.splitter or 'K-fold'} split",
                    "random_sample": ".sample()"}[s.kind]
            evidence = (f"datetime evidence on line {a.datetime_evidence[0][0]} ({a.datetime_evidence[0][1]})"
                        if strong else f"date-like column {date_cols[0]!r}")
            yield self.report(
                a, s.node,
                f"Random {what} on time-ordered data ({evidence}): the model trains on the future "
                "and is tested on the past, so scores look better than in production.",
                confidence=HIGH if strong else MEDIUM,
            )
