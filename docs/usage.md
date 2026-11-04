# Usage

## Commands

```bash
leakray check path/ [more paths]          # .py and .ipynb, recursively
leakray check nb.ipynb --format json      # or --format sarif --output leakray.sarif
leakray check . --select LK001,LK004      # only these rules (prefixes work: LK00)
leakray check . --ignore LK015
leakray check . --min-confidence medium   # low | medium | high | 0.0-1.0
leakray check . --fail-on error           # exit 1 only for errors (default: warning)
leakray rules                             # list rules
leakray explain LK003                     # why it leaks + how to fix
```

## Exit codes

| Code | Meaning |
|---|---|
| `0` | nothing at or above `--fail-on` |
| `1` | findings |
| `2` | usage or config error |

`--exit-zero` always exits 0.

## Output formats

`text` (default), `json`, and `sarif` for GitHub code scanning. Use `--output FILE` to write to a file.

## Notebooks

Code cells are joined in order (like "Run all"), IPython magics are ignored, and findings point to the cell: `nb.ipynb cell 4, line 2`.

## Configuration

In `pyproject.toml`:

```toml
[tool.leakray]
ignore = ["LK015"]
min-confidence = "medium"
fail-on = "warning"
exclude = ["scratch/*"]
per-file-ignores = { "notebooks/eda_*.ipynb" = ["LK014"] }
```

## Suppression

- One line: `# leakray: ignore[LK001]`. Without brackets, `# leakray: ignore` suppresses every rule on that line.
- Whole file: `# leakray: ignore-file[LK002]`.

Suppressed findings are counted in the summary line, so they stay visible.

## Tuning noise

Every finding has a confidence. Rules that rely on name hints (`*_id`, `date`, `target`) report medium or low confidence. Start with `--min-confidence medium` in CI and lower it when you review code by hand. Each [rule page](rules/index.md) lists that rule's false-positive cases.

False positives are bugs. [Report them](https://github.com/tpong-dev/leakray/issues/new?template=false_positive.yml).
