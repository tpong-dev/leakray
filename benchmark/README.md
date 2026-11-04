# leakray benchmark

A labelled set of small, realistic ML snippets used to measure how well leakray finds data leakage:
per-rule precision, recall and F1, plus file-level detection rates.

## Independence

The cases were written from one-line rule definitions only (what each LK rule is meant to catch),
without reading `src/leakray/rules/` or `tests/fixtures/`. They were not tuned to make the current
implementation pass. Misses and false alarms are expected and are listed in `RESULTS.md` for honest error
analysis. Please keep it that way: when a case fails, fix the analyser or argue the label, do not rewrite the
snippet until it passes.

## Layout

```
benchmark/
  cases/        one snippet per file (.py or .ipynb)
  run.py        runner + metrics
  results.json  machine-readable output (written by --write)
  RESULTS.md    human-readable output with FP / FN file lists (written by --write)
```

## Labelling

The first line of every `.py` case (the first line of the first code cell for a notebook) is the label:

```python
# expect: LK001            # one rule
# expect: LK001, LK009     # several leaks in one file
# expect: clean            # no leak; the analyser should stay silent
```

Rules for writing a label:

- Label every leak you put in the file on purpose, and nothing else.
- Unless the case is about LK015, call `.drop_duplicates()` on the loaded frame so the duplicate-row rule does
  not apply.
- Unless the case is about LK002, avoid date/time columns (`parse_dates`, `to_datetime`, `.dt`).
- Unless the case is about LK012, avoid `*_id` columns.
- Roughly 55% of cases are leaky and 45% clean. Every rule has at least 4 leaky cases and 3 clean "traps"
  (code that looks close to the leak but is correct: transform on test, Pipelines, constant `fillna`,
  `shift().rolling`, `TimeSeriesSplit`, `GroupKFold`, `eval_set` with a validation split, a single final test
  score, fitting after the split, and so on).
- Vary the style: different variable names, method chains, inline constructors, helper functions, notebooks.

File names follow `lk<rule>_<leaky|clean>_<nn>_<short_description>.py` (`multi_` / `misc_` for mixed cases).
The file name is for humans only; the runner reads the label line.

## Running

```bash
python benchmark/run.py                         # print the markdown table
python benchmark/run.py --min-confidence high   # count only findings with confidence >= 0.9 (low=0.3, medium=0.6)
python benchmark/run.py --write                 # also write results.json and RESULTS.md
```

The runner uses `leakray.engine.analyze_file` with default `Options`, counts each rule at most once per file,
and always exits 0 (it measures, it does not gate CI).

Metrics:

- Per rule: TP (labelled and found), FP (found, not labelled), FN (labelled, not found), precision, recall, F1.
- Micro: the same, summed over all rules.
- File level: a file counts as flagged if it has any finding, and as leaky if its label is not `clean`.

## Adding a case

1. Create `benchmark/cases/lkNNN_<leaky|clean>_<nn>_<what>.py` with the `# expect:` line first and the
   imports included (6 to 30 lines).
2. Follow the labelling rules above (drop duplicates, no dates, no `*_id` columns unless that is the point).
3. Run `python benchmark/run.py --write` and commit the updated `results.json` and `RESULTS.md` with the case.
