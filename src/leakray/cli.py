"""Command line: ``leakray check``, ``leakray rules``, ``leakray explain``."""

from __future__ import annotations

import argparse
import fnmatch
import io
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

from . import __version__
from .config import Config, ConfigError, find_pyproject, load_config
from .engine import FileResult, Options, analyze_file
from .models import Severity, parse_confidence
from .output import render_json, render_sarif, render_text
from .rules import all_rules, get_rule

SKIP_DIRS = {".git", ".venv", "venv", "env", ".env", "node_modules", "__pycache__", ".ipynb_checkpoints", "build",
             "dist", "site-packages", ".tox", ".nox", ".mypy_cache", ".ruff_cache", ".pytest_cache", "site"}


def iter_files(paths: Sequence[str], exclude: Sequence[str]) -> Iterator[Path]:
    seen: set[Path] = set()

    def excluded(p: Path) -> bool:
        posix = p.as_posix()
        return any(fnmatch.fnmatch(posix, g) or fnmatch.fnmatch(p.name, g) or fnmatch.fnmatch(posix, f"*/{g}")
                   for g in exclude)

    for raw in paths:
        p = Path(raw)
        if p.is_file():
            cands = [p]
        elif p.is_dir():
            cands = sorted(
                f for f in p.rglob("*")
                if f.suffix in {".py", ".ipynb"} and f.is_file()
                and not any(part in SKIP_DIRS for part in f.relative_to(p).parts[:-1])
            )
        else:
            raise FileNotFoundError(raw)
        for f in cands:
            if f not in seen and not excluded(f):
                seen.add(f)
                yield f


def _split_codes(v: str | None) -> list[str] | None:
    if v is None:
        return None
    return [c.strip().upper() for c in v.split(",") if c.strip()]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="leakray", description="Find data leakage in ML code (.py and .ipynb).")
    p.add_argument("--version", action="version", version=f"leakray {__version__}")
    sub = p.add_subparsers(dest="command")

    c = sub.add_parser("check", help="analyse files or folders")
    c.add_argument("paths", nargs="*", default=["."])
    c.add_argument("--format", choices=["text", "json", "sarif"], default="text")
    c.add_argument("--output", "-o", help="write the report to this file")
    c.add_argument("--select", help="comma-separated rule codes or prefixes to run (default: all)")
    c.add_argument("--ignore", help="comma-separated rule codes to skip")
    c.add_argument("--min-confidence", help="low | medium | high | a number in [0, 1]")
    c.add_argument("--fail-on", choices=["error", "warning", "info"], help="lowest severity that fails (default warning)")
    c.add_argument("--exit-zero", action="store_true", help="always exit 0")
    c.add_argument("--no-color", action="store_true")
    c.add_argument("--quiet", "-q", action="store_true", help="no snippets or fix hints")
    c.add_argument("--statistics", action="store_true", help="count findings per rule")
    c.add_argument("--exclude", action="append", default=[], help="glob to skip (repeatable)")
    c.add_argument("--config", help="path to a pyproject.toml")

    sub.add_parser("rules", help="list all rules")
    e = sub.add_parser("explain", help="explain one rule")
    e.add_argument("code")
    return p


def _cmd_rules(out: io.TextIOBase | object) -> int:
    for r in all_rules():
        print(f"{r.code}  {r.severity.value:<7}  {r.name:<36}  {r.summary}", file=out)  # type: ignore[arg-type]
    return 0


def _cmd_explain(code: str) -> int:
    try:
        r = get_rule(code)
    except KeyError:
        print(f"leakray: unknown rule {code!r}", file=sys.stderr)
        return 2
    print(f"{r.code} {r.name} ({r.severity.value}, default confidence {r.confidence})\n")
    print(r.summary + "\n")
    print("Fix: " + r.fix + "\n")
    print("Docs: " + r.doc_url)
    return 0


def _per_file(path: Path, cfg: Config) -> list[str]:
    posix = path.as_posix()
    out: list[str] = []
    for glob, codes in cfg.per_file_ignores.items():
        if fnmatch.fnmatch(posix, glob) or fnmatch.fnmatch(posix, f"*/{glob}"):
            out += codes
    return out


def _cmd_check(args: argparse.Namespace) -> int:
    try:
        cfg_path = Path(args.config) if args.config else find_pyproject(Path(args.paths[0]))
        cfg = load_config(cfg_path)
        select = _split_codes(args.select) if args.select else cfg.select
        ignore = (_split_codes(args.ignore) or []) + cfg.ignore
        min_conf = parse_confidence(args.min_confidence) if args.min_confidence else cfg.min_confidence
        fail_on = Severity(args.fail_on) if args.fail_on else cfg.fail_on
        files = list(iter_files(args.paths, [*cfg.exclude, *args.exclude]))
    except (ConfigError, ValueError) as exc:
        print(f"leakray: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"leakray: no such file or directory: {exc}", file=sys.stderr)
        return 2

    results: list[FileResult] = []
    for f in files:
        opts = Options(select=select, ignore=ignore + _per_file(f, cfg), min_confidence=min_conf)
        results.append(analyze_file(f, opts))

    stream = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout  # noqa: SIM115
    try:
        if args.format == "json":
            render_json(results, stream)
        elif args.format == "sarif":
            render_sarif(results, stream)
        else:
            color = not args.no_color and not args.output and sys.stdout.isatty()
            render_text(results, stream, color=color, quiet=args.quiet, statistics=args.statistics)
    finally:
        if args.output:
            stream.close()
    if args.format != "text":
        for r in results:
            if r.error:
                print(f"leakray: warning: {r.path}: {r.error}", file=sys.stderr)

    if args.exit_zero:
        return 0
    failing = [f for r in results for f in r.findings if f.severity.rank >= fail_on.rank]
    return 1 if failing else 0


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(errors="replace")  # type: ignore[union-attr]
        except Exception:  # pragma: no cover
            pass
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "check":
        return _cmd_check(args)
    if args.command == "rules":
        return _cmd_rules(sys.stdout)  # type: ignore[arg-type]
    if args.command == "explain":
        return _cmd_explain(args.code)
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
