"""Read ``[tool.leakray]`` from pyproject.toml."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

from .models import Severity, parse_confidence


class ConfigError(ValueError):
    pass


@dataclass
class Config:
    select: list[str] | None = None
    ignore: list[str] = field(default_factory=list)
    min_confidence: float = 0.0
    exclude: list[str] = field(default_factory=list)
    fail_on: Severity = Severity.WARNING
    per_file_ignores: dict[str, list[str]] = field(default_factory=dict)
    source: str | None = None


def _codes(value: Any, key: str) -> list[str]:
    if isinstance(value, str):
        value = [v for v in value.split(",") if v.strip()]
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"[tool.leakray] {key} must be a list of rule codes")
    return [v.strip().upper() for v in value]


def find_pyproject(start: Path) -> Path | None:
    cur = start.resolve()
    if cur.is_file():
        cur = cur.parent
    for d in (cur, *cur.parents):
        p = d / "pyproject.toml"
        if p.is_file():
            return p
    return None


def load_config(path: Path | None) -> Config:
    if path is None:
        return Config()
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    section = data.get("tool", {}).get("leakray", {})
    cfg = Config(source=str(path))
    for key, value in section.items():
        k = key.replace("_", "-")
        if k == "select":
            cfg.select = _codes(value, k)
        elif k in {"ignore", "extend-ignore"}:
            cfg.ignore += _codes(value, k)
        elif k == "min-confidence":
            try:
                cfg.min_confidence = parse_confidence(value)
            except ValueError as exc:
                raise ConfigError(f"bad min-confidence: {value!r}") from exc
        elif k == "exclude":
            cfg.exclude = list(value)
        elif k == "fail-on":
            try:
                cfg.fail_on = Severity(str(value).lower())
            except ValueError as exc:
                raise ConfigError(f"bad fail-on: {value!r}") from exc
        elif k == "per-file-ignores":
            if not isinstance(value, dict):
                raise ConfigError("per-file-ignores must be a table of glob = [codes]")
            cfg.per_file_ignores = {g: _codes(v, k) for g, v in value.items()}
        else:
            raise ConfigError(f"unknown [tool.leakray] key: {key}")
    return cfg
