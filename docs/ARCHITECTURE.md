# Architecture

leakray is a static analyser. It never runs your code.

```
src/leakray/
  models.py      Finding, Location, Severity, confidence levels (HIGH=0.9, MEDIUM=0.6, LOW=0.3)
  source.py      load .py / .ipynb -> SourceFile (text, ast tree, cell map for notebooks)
  dataflow.py    Analysis: one pass over the AST collecting facts (roles, splits, fits, calls)
  rules/base.py  Rule base class, @register, auto-discovery of rules/lk*.py
  rules/lkNNN_*.py  one rule per module
  engine.py      run rules, --select/--ignore, --min-confidence, inline suppressions
  config.py      [tool.leakray] in pyproject.toml
  cli.py         `leakray check`, `leakray rules`, `leakray explain`
  output/        text (rich), json, sarif renderers
```

## Pipeline

1. `load_source(path)` reads a file. For notebooks the code cells are
   concatenated in order; IPython magics (`%`, `!`) become `pass`, and
   `cell_map[line-1] = (cell_index, line_in_cell)` maps findings back.
2. `Analysis(src)` walks the module once, in source order, as one scope
   (like "Run all" in a notebook) and records facts.
3. Each registered rule's `check(analysis)` yields `Finding`s via `self.report(...)`.
4. The engine de-duplicates, filters by confidence, removes suppressed findings
   (`# leakray: ignore[LK001]` on the line, `# leakray: ignore-file[...]` anywhere).

## Facts available to rules (`Analysis`)

| Attribute / method | Meaning |
|---|---|
| `calls: list[CallInfo]` | every call, in order. `CallInfo.func` is the last name (`"fillna"`), `.name` is alias-resolved (`"sklearn.model_selection.train_test_split"`, `"df.fillna"`), `.receiver` is the variable before the dot, `.args`, `.kwargs`, `.kwarg_const(k)`, `.loop_depth`, `.stmt`, `.func_scope` |
| `calls_named(*funcs)` | calls whose `func` is in `funcs` |
| `splits: list[SplitEvent]` | `kind` is `train_test_split`, `cv_split` (`for tr, te in kf.split(X)`), `time_slice` (`df[df.date < "2020"]`), `positional` (`df[:n]`, `.iloc[:n]`), `random_sample` (`df.sample`), `subset`. `.inputs`, `.train`, `.test`, `.valid` names, `.splitter` (CV class), `.is_random`, `.call` |
| `first_split_line`, `has_split`, `split_after(line)`, `split_before(line)`, `is_before_any_split(line)` | split timing |
| `fits: list[FitCall]` | every `.fit/.fit_transform/.fit_resample/.fit_predict/.partial_fit`. `.estimator` (class short name or None), `.receiver`, `.data_args`, `.roles`, `.x_role`, `.role`, `.has_y`, `.call`, `.line` |
| `cv_evals: list[CallInfo]` | `cross_val_score`/`cross_validate`/... calls and `SearchCV.fit` calls |
| `assignments: list[Assignment]` | `.targets`, `.value`, `.role`, `.subscript_base`/`.subscript_key` for `df["c"] = ...` |
| `estimators: dict[name, list[EstimatorDef]]`, `estimator_class(name, line)`, `estimator_def(name, line)` | `scaler = StandardScaler()` -> `"StandardScaler"`; pipelines store `.steps` classes. `model = clf.fit(...)` aliases `model` to `clf`'s class |
| `role_at(name, line)`, `role_of(expr, line)` | data role: `FULL`, `TRAIN`, `VALID`, `TEST`, `UNKNOWN` (combine: TEST > VALID > TRAIN > FULL > UNKNOWN). Unassigned names fall back to name hints (`X_test` -> TEST) |
| `target_columns: dict[col, line]`, `target_vars: set[str]` | `y = df["price"]` -> `{"price"}`, `{"y"}` |
| `datetime_columns`, `datetime_evidence` | from `parse_dates=`, `pd.to_datetime(df["c"])`, `.dt`, `set_index("date")` |
| `string_constants` | every short string literal in the file |
| `resolve(node)`, `parent(node)`, `stmt_of(node)` | AST helpers |

Knowledge-base sets in `dataflow.py`: `PREPROCESSORS`, `SELECTORS`, `SAMPLERS`,
`TARGET_ENCODERS`, `SEARCH_CV`, `CV_SPLITTERS`, `CV_EVAL_FUNCS`, `PIPELINE_BUILDERS`.

## Known approximations

* One scope for the whole file; function bodies are walked in place.
* Roles are tracked per variable name, not per object; aliasing through
  containers (lists/dicts) is not followed.
* Name hints (`train`, `test`, `val`) are used when the dataflow has nothing better.
