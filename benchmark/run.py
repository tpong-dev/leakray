"""Run leakray over the labelled benchmark cases and report precision / recall per rule.

Usage:
    python benchmark/run.py                       # print a markdown table
    python benchmark/run.py --min-confidence high # only count findings with confidence >= 0.9
    python benchmark/run.py --write               # also write benchmark/results.json and benchmark/RESULTS.md

Always exits 0: the benchmark measures the analyser, it is not a gate.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASES_DIR = HERE / "cases"
CONFIDENCE_LEVELS = {"low": 0.3, "medium": 0.6, "high": 0.9}
EXPECT_RE = re.compile(r"^\s*#\s*expect:\s*(.+?)\s*$", re.IGNORECASE)


@dataclass
class CaseResult:
    name: str
    expected: set[str]
    predicted: set[str]
    error: str | None = None


@dataclass
class RuleStats:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    fp_files: list[str] = field(default_factory=list)
    fn_files: list[str] = field(default_factory=list)


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = safe_div(tp, tp + fp)
    r = safe_div(tp, tp + fn)
    return {"precision": p, "recall": r, "f1": safe_div(2 * p * r, p + r)}


def parse_expect(line: str) -> set[str] | None:
    """Parse '# expect: LK001, LK009' or '# expect: clean'. Returns None when the line is not a label."""
    m = EXPECT_RE.match(line)
    if not m:
        return None
    value = m.group(1).strip()
    if value.lower() == "clean":
        return set()
    return {code.strip().upper() for code in value.split(",") if code.strip()}


def read_label(path: Path) -> set[str] | None:
    """Label is the first line of a .py file, or the first line of the first code cell of a notebook."""
    if path.suffix == ".ipynb":
        nb = json.loads(path.read_text(encoding="utf-8"))
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "code":
                src = cell.get("source", "")
                text = "".join(src) if isinstance(src, list) else src
                return parse_expect(text.splitlines()[0] if text else "")
        return None
    with path.open(encoding="utf-8") as fh:
        return parse_expect(fh.readline())


def compute_metrics(results: list[CaseResult]) -> dict:
    """Per-rule TP/FP/FN with precision/recall/F1, micro totals and file-level detection metrics.

    A rule is counted at most once per file (several findings of the same rule in one file = one prediction).
    File level: a file is 'flagged' when it has any finding; it is 'leaky' when its label is not clean.
    """
    rules: dict[str, RuleStats] = {}
    file_tp = file_fp = file_fn = file_tn = 0
    for res in results:
        for code in res.expected | res.predicted:
            st = rules.setdefault(code, RuleStats())
            if code in res.expected and code in res.predicted:
                st.tp += 1
            elif code in res.predicted:
                st.fp += 1
                st.fp_files.append(res.name)
            else:
                st.fn += 1
                st.fn_files.append(res.name)
        leaky, flagged = bool(res.expected), bool(res.predicted)
        if leaky and flagged:
            file_tp += 1
        elif flagged:
            file_fp += 1
        elif leaky:
            file_fn += 1
        else:
            file_tn += 1

    per_rule = {}
    for code in sorted(rules):
        st = rules[code]
        per_rule[code] = {
            "tp": st.tp, "fp": st.fp, "fn": st.fn, **prf(st.tp, st.fp, st.fn),
            "fp_files": sorted(st.fp_files), "fn_files": sorted(st.fn_files),
        }
    tp = sum(s.tp for s in rules.values())
    fp = sum(s.fp for s in rules.values())
    fn = sum(s.fn for s in rules.values())
    return {
        "per_rule": per_rule,
        "micro": {"tp": tp, "fp": fp, "fn": fn, **prf(tp, fp, fn)},
        "file_level": {
            "tp": file_tp, "fp": file_fp, "fn": file_fn, "tn": file_tn,
            "precision": safe_div(file_tp, file_tp + file_fp),
            "recall": safe_div(file_tp, file_tp + file_fn),
        },
    }


def run_cases(cases_dir: Path, min_confidence: float | None) -> list[CaseResult]:
    from leakray.engine import Options, analyze_file

    opts = Options() if min_confidence is None else Options(min_confidence=min_confidence)
    results = []
    for path in sorted([*cases_dir.glob("*.py"), *cases_dir.glob("*.ipynb")]):
        expected = read_label(path)
        if expected is None:
            print(f"warning: {path.name} has no '# expect:' label, skipped", file=sys.stderr)
            continue
        fr = analyze_file(path, opts)
        error = str(fr.error) if fr.error else None
        predicted = set() if error else {f.rule for f in fr.findings}
        results.append(CaseResult(path.name, expected, predicted, error))
    return results


def markdown_table(metrics: dict) -> str:
    lines = ["| Rule | TP | FP | FN | Precision | Recall | F1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for code, m in metrics["per_rule"].items():
        lines.append(
            f"| {code} | {m['tp']} | {m['fp']} | {m['fn']} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} |"
        )
    mi = metrics["micro"]
    lines.append(
        f"| **micro** | {mi['tp']} | {mi['fp']} | {mi['fn']} | "
        f"**{mi['precision']:.2f}** | **{mi['recall']:.2f}** | **{mi['f1']:.2f}** |"
    )
    fl = metrics["file_level"]
    lines.append("")
    lines.append(
        f"File level (any finding vs labelled leaky): precision {fl['precision']:.2f}, recall {fl['recall']:.2f} "
        f"(TP {fl['tp']}, FP {fl['fp']}, FN {fl['fn']}, TN {fl['tn']})"
    )
    return "\n".join(lines)


def results_markdown(metrics: dict, results: list[CaseResult], version: str, level: str | None) -> str:
    n_leaky = sum(1 for r in results if r.expected)
    out = [
        "# leakray benchmark results",
        "",
        f"- Date: {dt.date.today().isoformat()}",
        f"- leakray version: {version}",
        f"- Cases: {len(results)} ({n_leaky} leaky, {len(results) - n_leaky} clean)",
        f"- Confidence filter: {level or 'none (default Options)'}",
        "",
        "Generated by `python benchmark/run.py --write`. Do not edit by hand.",
        "",
        "## Metrics",
        "",
        markdown_table(metrics),
        "",
        "## Error analysis",
        "",
        "False positives (rule fired, not labelled) and false negatives (labelled, rule did not fire).",
        "",
    ]
    any_err = False
    for code, m in metrics["per_rule"].items():
        if not m["fp_files"] and not m["fn_files"]:
            continue
        any_err = True
        out.append(f"### {code}")
        out.append("")
        if m["fp_files"]:
            out.append("- FP: " + ", ".join(f"`{f}`" for f in m["fp_files"]))
        if m["fn_files"]:
            out.append("- FN: " + ", ".join(f"`{f}`" for f in m["fn_files"]))
        out.append("")
    if not any_err:
        out.append("No errors.")
        out.append("")
    errors = [r for r in results if r.error]
    if errors:
        out.append("## Analysis errors")
        out.append("")
        out.extend(f"- `{r.name}`: {r.error}" for r in errors)
        out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--min-confidence", choices=sorted(CONFIDENCE_LEVELS), default=None)
    parser.add_argument("--write", action="store_true", help="write results.json and RESULTS.md")
    parser.add_argument("--cases", type=Path, default=CASES_DIR)
    args = parser.parse_args(argv)

    try:
        from leakray import __version__ as version
    except ImportError:
        version = "unknown"
    threshold = CONFIDENCE_LEVELS.get(args.min_confidence) if args.min_confidence else None
    results = run_cases(args.cases, threshold)
    metrics = compute_metrics(results)
    n_leaky = sum(1 for r in results if r.expected)
    print(f"leakray {version}: {len(results)} cases ({n_leaky} leaky, {len(results) - n_leaky} clean)\n")
    print(markdown_table(metrics))
    for r in results:
        if r.error:
            print(f"error: {r.name}: {r.error}", file=sys.stderr)

    if args.write:
        payload = {
            "date": dt.date.today().isoformat(),
            "leakray_version": version,
            "min_confidence": args.min_confidence,
            "cases": len(results),
            "leaky_cases": n_leaky,
            **metrics,
            "files": [
                {"name": r.name, "expected": sorted(r.expected), "predicted": sorted(r.predicted), "error": r.error}
                for r in results
            ],
        }
        (HERE / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        (HERE / "RESULTS.md").write_text(
            results_markdown(metrics, results, version, args.min_confidence), encoding="utf-8"
        )
        print(f"\nwrote {HERE / 'results.json'} and {HERE / 'RESULTS.md'}")
    return 0


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # the benchmark never fails the caller
        print(f"benchmark error: {exc}", file=sys.stderr)
    sys.exit(0)
