"""Rule base class and registry.

Every rule lives in its own module ``leakray/rules/lkNNN_<slug>.py`` and is
registered with the ``@register`` decorator. Modules are discovered
automatically, so adding a rule never requires editing a shared file.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
from collections.abc import Iterator
from typing import ClassVar

from ..dataflow import Analysis
from ..models import MEDIUM, Finding, Location, Severity

_REGISTRY: dict[str, type[Rule]] = {}


class Rule:
    code: ClassVar[str]  # "LK001"
    name: ClassVar[str]  # "preprocessing-fit-before-split" (kebab-case)
    summary: ClassVar[str]  # one line, shown in rule tables
    severity: ClassVar[Severity] = Severity.WARNING
    confidence: ClassVar[float] = MEDIUM  # default confidence of findings
    fix: ClassVar[str]  # default fix hint
    doc_url: ClassVar[str] = ""  # filled in by register()

    def check(self, a: Analysis) -> Iterator[Finding]:  # pragma: no cover - abstract
        raise NotImplementedError
        yield

    # Helper used by every rule to build a Finding with a correct location.
    def report(
        self,
        a: Analysis,
        node: ast.AST,
        message: str,
        *,
        confidence: float | None = None,
        fix: str | None = None,
        severity: Severity | None = None,
        **extra: object,
    ) -> Finding:
        line = getattr(node, "lineno", 1)
        col = getattr(node, "col_offset", 0)
        end_line = getattr(node, "end_lineno", None)
        end_col = getattr(node, "end_col_offset", None)
        cell, cell_line = a.src.map_line(line)
        loc = Location(
            path=a.src.path, line=line, col=col, end_line=end_line, end_col=end_col,
            cell=cell, cell_line=cell_line,
        )
        return Finding(
            rule=self.code,
            name=self.name,
            message=message,
            severity=severity or self.severity,
            confidence=self.confidence if confidence is None else confidence,
            location=loc,
            fix=fix or self.fix,
            snippet=a.src.snippet(line, end_line),
            extra=dict(extra),
        )


DOCS_BASE = "https://tpong-dev.github.io/leakray/rules/"


def register(cls: type[Rule]) -> type[Rule]:
    if cls.code in _REGISTRY and _REGISTRY[cls.code] is not cls:
        raise ValueError(f"duplicate rule code {cls.code}")
    cls.doc_url = f"{DOCS_BASE}{cls.code}/"
    _REGISTRY[cls.code] = cls
    return cls


def _discover() -> None:
    from . import __path__ as pkg_path

    for mod in pkgutil.iter_modules(pkg_path):
        if mod.name.lower().startswith("lk"):
            importlib.import_module(f"{__package__}.{mod.name}")


def all_rules() -> list[type[Rule]]:
    """All registered rule classes, sorted by code."""
    if not _REGISTRY:
        _discover()
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


def get_rule(code: str) -> type[Rule]:
    all_rules()
    return _REGISTRY[code.upper()]
