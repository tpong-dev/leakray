# Integrations

## pre-commit

```yaml
repos:
  - repo: https://github.com/tpong-dev/leakray
    rev: v0.1.0
    hooks:
      - id: leakray            # or leakray-errors-only
```

Use `leakray-errors-only` to fail only on `error` findings.

## GitHub Action

Findings appear as code-scanning alerts on pull requests (free for public repositories).

```yaml
permissions: { contents: read, security-events: write }
steps:
  - uses: actions/checkout@v4
  - uses: tpong-dev/leakray@v0.1.0
    with: { path: ".", fail-on: "error" }
```

## SARIF without the Action

```bash
leakray check . --format sarif --output leakray.sarif --exit-zero
```

Upload the file with `github/codeql-action/upload-sarif` or open it in any SARIF viewer.

## Jupyter notebooks

`leakray check` reads `.ipynb` files directly, so the same commands, hooks and Action cover notebooks. Findings are reported per cell and line.

## JSON

`--format json` gives a machine-readable list of findings for your own tooling.
