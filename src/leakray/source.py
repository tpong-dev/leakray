"""Load .py and .ipynb files into one parsed source with line mapping."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

NL = "\n"

# IPython magics / shell escapes that are not valid Python.
_MAGIC_RE = re.compile(r"^\s*([%!?]|\w+\s*=\s*[%!])")


@dataclass
class SourceFile:
    path: str
    text: str
    tree: ast.Module
    # For notebooks: one entry per line of ``text`` -> (cell_index, line_in_cell).
    cell_map: list[tuple[int, int]] | None = None
    lines: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.lines:
            self.lines = self.text.splitlines()

    @property
    def is_notebook(self) -> bool:
        return self.cell_map is not None

    def map_line(self, line: int) -> tuple[int | None, int | None]:
        """Map a 1-based line in ``text`` to (cell, line-in-cell) for notebooks."""
        if self.cell_map is None or not (1 <= line <= len(self.cell_map)):
            return None, None
        return self.cell_map[line - 1]

    def line_text(self, line: int) -> str:
        if 1 <= line <= len(self.lines):
            return self.lines[line - 1]
        return ""

    def snippet(self, line: int, end_line: int | None = None) -> str:
        end = end_line or line
        end = min(end, line + 4)  # keep snippets short
        return "\n".join(self.line_text(i) for i in range(line, end + 1))


def _clean_cell_line(line: str) -> str:
    """Replace IPython-only syntax with an equivalent-length no-op."""
    if _MAGIC_RE.match(line):
        indent = line[: len(line) - len(line.lstrip())]
        return f"{indent}pass  # {line.strip()}"
    return line


def notebook_code(raw: str) -> tuple[str, list[tuple[int, int]]]:
    """Concatenate code cells; return (text, cell_map).

    Uses plain json so it works even without nbformat; nbformat is used when
    available to validate/upgrade old notebook versions.
    """
    try:
        import nbformat

        nb = nbformat.reads(raw, as_version=4)
        cells = [(c.get("cell_type"), c.get("source", "")) for c in nb.cells]
    except Exception:
        data = json.loads(raw)
        cells = []
        for c in data.get("cells", []):
            src = c.get("source", "")
            if isinstance(src, list):
                src = "".join(src)
            cells.append((c.get("cell_type"), src))

    out_lines: list[str] = []
    cell_map: list[tuple[int, int]] = []
    for idx, (ctype, src) in enumerate(cells):
        if ctype != "code":
            continue
        if isinstance(src, list):
            src = "".join(src)
        lines = src.splitlines() or [""]
        if lines and lines[0].lstrip().startswith("%%"):
            # Cell magic (%%time, %%capture, %%bash...). Keep Python ones.
            first = lines[0].strip()
            if first.startswith(("%%time", "%%capture", "%%timeit")):
                lines = ["pass  # " + first, *lines[1:]]
            else:
                lines = ["# " + ln for ln in lines]
        for i, ln in enumerate(lines, start=1):
            out_lines.append(_clean_cell_line(ln))
            cell_map.append((idx, i))
    return "\n".join(out_lines) + "\n", cell_map


def _is_python_notebook(raw: str) -> bool:
    try:
        meta = json.loads(raw).get("metadata", {})
    except (ValueError, AttributeError):
        return True  # let the normal path report the problem
    lang = (meta.get("kernelspec", {}) or {}).get("language") or (meta.get("language_info", {}) or {}).get("name")
    return lang is None or str(lang).lower().startswith("python")


def _blank_bad_cells(text: str, cell_map: list[tuple[int, int]]) -> str:
    """Replace every line of a cell that does not parse on its own with ``pass``.

    One broken cell (unusual magic, invalid syntax, other language) then costs only that cell,
    not the whole notebook.
    """
    lines = text.splitlines()
    try:
        ast.parse(text)
        return text
    except SyntaxError:
        pass
    by_cell: dict[int, list[int]] = {}
    for i, (cell, _) in enumerate(cell_map):
        by_cell.setdefault(cell, []).append(i)
    for idxs in by_cell.values():
        src = NL.join(lines[i] for i in idxs)
        try:
            ast.parse(src)
        except SyntaxError:
            for i in idxs:
                lines[i] = "pass"
    return NL.join(lines) + NL


def _parse_lenient(text: str, path: str) -> ast.Module:
    """Parse; if a notebook cell has a syntax error, blank that line and retry."""
    lines = text.splitlines()
    for _ in range(50):
        try:
            return ast.parse("\n".join(lines) + "\n", filename=path)
        except SyntaxError as exc:
            if exc.lineno is None or not (1 <= exc.lineno <= len(lines)):
                raise
            bad = exc.lineno - 1
            indent = lines[bad][: len(lines[bad]) - len(lines[bad].lstrip())]
            lines[bad] = f"{indent}pass"
    raise SyntaxError(f"could not parse {path}")


def load_source(path: str | Path) -> SourceFile:
    """Read and parse a .py or .ipynb file. Raises SyntaxError / ValueError."""
    p = Path(path)
    raw = p.read_text(encoding="utf-8", errors="replace")
    if p.suffix == ".ipynb":
        if not raw.strip() or not _is_python_notebook(raw):
            # empty file, or an R/Julia/... notebook: nothing for us to analyse
            return SourceFile(path=str(p), text="", tree=ast.Module(body=[], type_ignores=[]), cell_map=[])
        text, cell_map = notebook_code(raw)
        text = _blank_bad_cells(text, cell_map)
        tree = _parse_lenient(text, str(p))
        return SourceFile(path=str(p), text=text, tree=tree, cell_map=cell_map)
    tree = ast.parse(raw, filename=str(p))
    return SourceFile(path=str(p), text=raw, tree=tree)


def source_from_string(text: str, path: str = "<string>") -> SourceFile:
    """Parse Python source given as a string (used heavily in tests)."""
    return SourceFile(path=path, text=text, tree=ast.parse(text, filename=path))
