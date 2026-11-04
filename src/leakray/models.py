"""Core data model: findings, severities, confidence levels, locations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

    @property
    def rank(self) -> int:
        return {"info": 0, "warning": 1, "error": 2}[self.value]


# Confidence is a float in [0, 1]. These named levels are what rules use and
# what `--min-confidence low|medium|high` maps to.
HIGH = 0.9
MEDIUM = 0.6
LOW = 0.3

CONFIDENCE_LEVELS: dict[str, float] = {"low": LOW, "medium": MEDIUM, "high": HIGH}


def confidence_label(value: float) -> str:
    if value >= HIGH:
        return "high"
    if value >= MEDIUM:
        return "medium"
    return "low"


def parse_confidence(value: str | float) -> float:
    """Accept 'low'/'medium'/'high' or a number in [0, 1]."""
    if isinstance(value, (int, float)):
        return float(value)
    v = value.strip().lower()
    if v in CONFIDENCE_LEVELS:
        return CONFIDENCE_LEVELS[v]
    return float(v)


@dataclass(frozen=True)
class Location:
    """Where a finding is.

    ``line``/``col`` are 1-based line and 0-based column in the *analysed
    source*: for a ``.py`` file that is the file itself; for a notebook it is
    the concatenation of code cells. ``cell`` (0-based index among all cells)
    and ``cell_line`` (1-based line inside that cell) are set for notebooks.
    """

    path: str
    line: int
    col: int = 0
    end_line: int | None = None
    end_col: int | None = None
    cell: int | None = None
    cell_line: int | None = None

    @property
    def display_line(self) -> int:
        """Line the user should look at (cell-relative for notebooks)."""
        return self.cell_line if self.cell_line is not None else self.line

    def __str__(self) -> str:
        if self.cell is not None:
            return f"{self.path}:cell[{self.cell}]:{self.cell_line}:{self.col + 1}"
        return f"{self.path}:{self.line}:{self.col + 1}"


@dataclass
class Finding:
    rule: str  # e.g. "LK001"
    name: str  # e.g. "preprocessing-fit-before-split"
    message: str
    severity: Severity
    confidence: float
    location: Location
    fix: str
    snippet: str = ""  # the offending source line(s)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def confidence_label(self) -> str:
        return confidence_label(self.confidence)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        d["confidence_label"] = self.confidence_label
        return d

    def sort_key(self) -> tuple[str, int, int, str]:
        return (self.location.path, self.location.line, self.location.col, self.rule)
