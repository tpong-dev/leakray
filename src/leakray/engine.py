"""Run rules over files, apply selection, confidence filter and suppressions."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .dataflow import Analysis
from .models import Finding, Location, Severity
from .rules import all_rules
from .source import SourceFile, load_source, source_from_string

# ``# leakray: ignore`` or ``# leakray: ignore[LK001, LK004]`` (same line)
_IGNORE_RE = re.compile(r"#\s*leakray\s*:\s*ignore(?!-file)(?:\[([A-Za-z0-9_,\s]+)\])?", re.I)
# ``# leakray: ignore-file`` / ``# leakray: ignore-file[LK002]`` (anywhere)
_IGNORE_FILE_RE = re.compile(r"#\s*leakray\s*:\s*ignore-file(?:\[([A-Za-z0-9_,\s]+)\])?", re.I)

ALL = "*"


@dataclass
class Options:
    """What to run. Built by the CLI from flags + pyproject config."""

    select: list[str] | None = None  # rule codes or prefixes ("LK00"); None = all
    ignore: list[str] = field(default_factory=list)
    min_confidence: float = 0.0


@dataclass
class FileResult:
    path: str
    findings: list[Finding] = field(default_factory=list)
    error: str | None = None  # parse error etc.
    suppressed: int = 0


def _matches(code: str, patterns: Iterable[str]) -> bool:
    code = code.upper()
    return any(code.startswith(p.upper()) or p == ALL for p in patterns)


def rule_enabled(code: str, opts: Options) -> bool:
    if opts.select is not None and not _matches(code, opts.select):
        return False
    return not _matches(code, opts.ignore)


def _codes(group: str | None) -> set[str]:
    if not group:
        return {ALL}
    return {c.strip().upper() for c in group.split(",") if c.strip()}


def suppressions(src: SourceFile) -> tuple[dict[int, set[str]], set[str]]:
    """Return (line -> suppressed codes, file-level suppressed codes)."""
    per_line: dict[int, set[str]] = {}
    file_level: set[str] = set()
    for i, text in enumerate(src.lines, start=1):
        if "leakray" not in text.lower():
            continue
        m = _IGNORE_FILE_RE.search(text)
        if m:
            file_level |= _codes(m.group(1))
            continue
        m = _IGNORE_RE.search(text)
        if m:
            per_line.setdefault(i, set()).update(_codes(m.group(1)))
    return per_line, file_level


def _is_suppressed(f: Finding, per_line: dict[int, set[str]], file_level: set[str]) -> bool:
    if ALL in file_level or f.rule in file_level:
        return True
    start = f.location.line
    end = f.location.end_line or start
    for ln in range(start, end + 1):
        codes = per_line.get(ln)
        if codes and (ALL in codes or f.rule in codes):
            return True
    return False


def analyze_source(src: SourceFile, opts: Options | None = None) -> FileResult:
    opts = opts or Options()
    result = FileResult(path=src.path)
    a = Analysis(src)
    per_line, file_level = suppressions(src)
    seen: set[tuple[str, int, int]] = set()
    for rule_cls in all_rules():
        if not rule_enabled(rule_cls.code, opts):
            continue
        rule = rule_cls()
        try:
            found = list(rule.check(a))
        except Exception as exc:  # a buggy rule must never crash the run
            result.error = f"internal error in {rule_cls.code}: {exc!r}"
            continue
        for f in found:
            key = (f.rule, f.location.line, f.location.col)
            if key in seen:
                continue
            seen.add(key)
            if f.confidence < opts.min_confidence:
                continue
            if _is_suppressed(f, per_line, file_level):
                result.suppressed += 1
                continue
            result.findings.append(f)
    result.findings.sort(key=Finding.sort_key)
    return result


def analyze_file(path: str | Path, opts: Options | None = None) -> FileResult:
    try:
        src = load_source(path)
    except (SyntaxError, ValueError, UnicodeDecodeError, OSError) as exc:
        return FileResult(path=str(path), error=f"could not parse: {exc}")
    return analyze_source(src, opts)


def analyze_string(code: str, opts: Options | None = None, path: str = "<string>") -> list[Finding]:
    """Convenience for tests: analyse a code string and return findings."""
    return analyze_source(source_from_string(code, path), opts).findings


def parse_error_finding(path: str, message: str) -> Finding:
    return Finding(
        rule="LK000", name="parse-error", message=message, severity=Severity.INFO,
        confidence=1.0, location=Location(path=path, line=1), fix="Fix the syntax error.",
    )
