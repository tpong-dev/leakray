"""Render results as text (rich), JSON or SARIF 2.1.0."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import IO, Any

from . import __version__
from .engine import FileResult
from .models import Finding, Severity
from .rules import all_rules

_SEV_STYLE = {Severity.ERROR: "bold red", Severity.WARNING: "yellow", Severity.INFO: "cyan"}
_SARIF_LEVEL = {Severity.ERROR: "error", Severity.WARNING: "warning", Severity.INFO: "note"}


def summary(results: list[FileResult]) -> dict[str, Any]:
    findings = [f for r in results for f in r.findings]
    return {
        "files": len(results),
        "findings": len(findings),
        "errors": sum(f.severity is Severity.ERROR for f in findings),
        "warnings": sum(f.severity is Severity.WARNING for f in findings),
        "info": sum(f.severity is Severity.INFO for f in findings),
        "suppressed": sum(r.suppressed for r in results),
        "parse_errors": sum(r.error is not None for r in results),
    }


def _where(f: Finding) -> str:
    loc = f.location
    if loc.cell is not None:
        return f"{loc.path} cell {loc.cell + 1}, line {loc.cell_line}:{loc.col + 1}"
    return f"{loc.path}:{loc.line}:{loc.col + 1}"


def render_text(results: list[FileResult], stream: IO[str], color: bool = True, quiet: bool = False,
                statistics: bool = False) -> None:
    from rich.console import Console
    from rich.markup import escape

    console = Console(file=stream, force_terminal=color, no_color=not color, highlight=False,
                      soft_wrap=True, emoji=False)
    for r in results:
        for f in r.findings:
            style = _SEV_STYLE[f.severity]
            console.print(f"[bold]{escape(_where(f))}[/bold] [{style}]{f.rule}[/{style}] "
                          f"[dim]{f.severity.value}, {f.confidence_label} confidence[/dim]")
            console.print(f"  {escape(f.message)}")
            if not quiet:
                first = f.location.display_line
                for i, text in enumerate(f.snippet.splitlines()):
                    console.print(f"  [dim]{first + i:>4} |[/dim] {escape(text)}")
                console.print(f"  [green]fix:[/green] {escape(f.fix)}")
                console.print(f"  [dim]docs: https://tpong-dev.github.io/leakray/rules/{f.rule}/[/dim]")
            console.print()
    for r in results:
        if r.error:
            console.print(f"[dim]warning: {escape(r.path)}: {escape(r.error)}[/dim]")
    s = summary(results)
    if statistics and s["findings"]:
        counts: dict[str, int] = {}
        for r in results:
            for f in r.findings:
                counts[f.rule] = counts.get(f.rule, 0) + 1
        for code in sorted(counts):
            console.print(f"{counts[code]:>5}  {code}")
    if s["findings"]:
        console.print(f"[bold]{s['findings']} finding(s)[/bold] ({s['errors']} error, {s['warnings']} warning, "
                      f"{s['info']} info) in {s['files']} file(s); {s['suppressed']} suppressed.")
    else:
        console.print(f"[green]No leakage found[/green] in {s['files']} file(s); {s['suppressed']} suppressed.")


def render_json(results: list[FileResult], stream: IO[str]) -> None:
    out = {
        "version": __version__,
        "findings": [f.to_dict() for r in results for f in r.findings],
        "summary": summary(results),
        "errors": [{"path": r.path, "error": r.error} for r in results if r.error],
    }
    json.dump(out, stream, indent=2)
    stream.write("\n")


def _rel_uri(path: str) -> str:
    p = Path(path)
    try:
        p = p.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        pass
    return p.as_posix()


def _precision(conf: float) -> str:
    return "high" if conf >= 0.9 else "medium" if conf >= 0.6 else "low"


def to_sarif(results: list[FileResult]) -> dict[str, Any]:
    rules = all_rules()
    index = {r.code: i for i, r in enumerate(rules)}
    driver_rules = [
        {
            "id": r.code,
            "name": r.name,
            "shortDescription": {"text": r.summary},
            "fullDescription": {"text": r.summary},
            "help": {"text": f"{r.summary}\n\nFix: {r.fix}", "markdown": f"**{r.summary}**\n\nFix: {r.fix}\n\n"
                     f"[Documentation]({r.doc_url})"},
            "helpUri": r.doc_url,
            "defaultConfiguration": {"level": _SARIF_LEVEL[r.severity]},
            "properties": {"tags": ["ml", "data-leakage"], "precision": _precision(r.confidence)},
        }
        for r in rules
    ]
    sarif_results = []
    for res in results:
        for f in res.findings:
            loc = f.location
            msg = f.message + f" Fix: {f.fix}"
            if loc.cell is not None:
                msg = f"[cell {loc.cell + 1}, line {loc.cell_line}] " + msg
                region = {"startLine": 1}
            else:
                region = {"startLine": loc.line, "startColumn": loc.col + 1}
                if loc.end_line:
                    region["endLine"] = loc.end_line
                if loc.end_col is not None and (loc.end_line or loc.line) == loc.line:
                    region["endColumn"] = loc.end_col + 1
            uri = _rel_uri(loc.path)
            fp = hashlib.sha256(f"{f.rule}|{uri}|{f.snippet.strip()}".encode()).hexdigest()[:32]
            item: dict[str, Any] = {
                "ruleId": f.rule,
                "level": _SARIF_LEVEL[f.severity],
                "message": {"text": msg},
                "locations": [{"physicalLocation": {
                    "artifactLocation": {"uri": uri, "uriBaseId": "%SRCROOT%"}, "region": region}}],
                "partialFingerprints": {"leakray/v1": fp},
                "properties": {"confidence": f.confidence},
            }
            if f.rule in index:
                item["ruleIndex"] = index[f.rule]
            sarif_results.append(item)
    return {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [{
            "tool": {"driver": {"name": "leakray", "version": __version__,
                                "informationUri": "https://github.com/tpong-dev/leakray", "rules": driver_rules}},
            "results": sarif_results,
        }],
    }


def render_sarif(results: list[FileResult], stream: IO[str]) -> None:
    json.dump(to_sarif(results), stream, indent=2)
    stream.write("\n")
