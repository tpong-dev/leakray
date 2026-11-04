from __future__ import annotations

import pytest

from leakray.rules._util import explicit_date_column


@pytest.mark.parametrize("name", ["date", "Timestamp", "ds", "time", "order_date", "date_joined", "created_at"])
def test_explicit_date_column_accepts(name: str) -> None:
    assert explicit_date_column(name)


@pytest.mark.parametrize("name", ["spend_per_month", "hour_of_day", "visits_per_week", "dwell_time", "update"])
def test_explicit_date_column_rejects(name: str) -> None:
    assert not explicit_date_column(name)
