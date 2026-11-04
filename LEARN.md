# Learning leakray

This file is for me, the author, as a learner. It explains in simple words how the tool works, what each kind of leakage is, and what I must be able to explain in an interview. It ends with a quiz and three hands-on exercises.

## 1. The big picture in one paragraph

leakray reads Python code **without running it**. It turns the code into a tree (an AST), walks that tree once from top to bottom, and writes down facts: which variables hold the full dataset, which hold only the training or test part, where the split happens, and which `.fit()` call saw which data. Then each rule asks one question about those facts, for example "was a scaler fitted on full data, with a split later?". If the answer is yes, it reports a finding with a fix.

## 2. What an AST is

Python code is text. Before Python runs it, it **parses** the text into a tree. Each node in the tree is a piece of syntax: an assignment, a function call, a name, a number.

```python
X = scaler.fit_transform(df)
```

becomes roughly:

```
Assign
├── targets: Name("X")
└── value: Call
    ├── func: Attribute(value=Name("scaler"), attr="fit_transform")
    └── args: [Name("df")]
```

You can see this yourself:

```python
import ast
print(ast.dump(ast.parse("X = scaler.fit_transform(df)"), indent=2))
```

Why use the AST instead of regex on text? The tree already knows what is a call, what is an argument, and which line it is on. A regex would confuse `fit` in a comment with a real `.fit()` call, and it breaks on line breaks and nesting.

`ast.NodeVisitor` lets you write `visit_Call`, `visit_Assign` and so on. Python calls them for each node of that type as it walks the tree. leakray's walker is `_Walker` in [src/leakray/dataflow.py](src/leakray/dataflow.py).

## 3. What dataflow analysis is (the leakray version)

**Dataflow analysis** answers "what values can this variable hold at this point in the program?". Real compilers do this very precisely. leakray does a simple version: it tracks a **role** for each variable name.

| Role | Meaning | Example |
|---|---|---|
| `full` | the whole dataset, before any split | `df = pd.read_csv(...)` |
| `train` | only the training side | `X_train` from `train_test_split` |
| `valid` | a validation side | `X_val` |
| `test` | only the test side | `X_test`, `X.iloc[test_idx]` |
| `unknown` | anything else | `model = LogisticRegression()` |

Roles **flow** through assignments. If a new variable is built from other variables, it takes the "strongest" of their roles, in the order `test > valid > train > full > unknown`. So:

- `X = df.drop(columns=["y"])` → `full` (built from `df`)
- `X_test_s = scaler.transform(X_test)` → `test`
- `both = pd.concat([X_train, X_test])` → `test` (it contains test rows)

It is **flow-sensitive**: the role is looked up "as of" a line. That matters because people reuse names: `X = scaler.fit_transform(X)` reads the *old* `X` on that line.

Where roles come from:

1. **Loaders**: `read_csv`, `load_iris`, ... make `full` data, including inside a chain like `pd.read_csv(...).drop_duplicates()`.
2. **Splits**: `a, b, c, d = train_test_split(X, y)` gives `a, c` train and `b, d` test, based on position. A for-loop over `kf.split(X)` gives the two loop variables train and test roles.
3. **Name hints**: `X_test`, `train_df`, `va` are guessed from their names when nothing better is known. This is the weakest source and the main cause of false positives (see [docs/dogfood.md](docs/dogfood.md)).

Besides roles, one pass also records:

- every call, with import aliases resolved (`pd.read_csv` → `pandas.read_csv`)
- every **split event** and its line
- every `.fit`, `.fit_transform` and `.fit_resample` call, with the **estimator class**. `scaler = StandardScaler()` is remembered, so `scaler.fit(X)` is known to be a StandardScaler fit.
- the **target column**: `y = df["price"]` means `"price"` is the target
- **datetime evidence**: `parse_dates=`, `pd.to_datetime`, `.dt`

### Honest limits (I must be able to say these)

- One scope for the whole file. Functions are walked in place, and calls into other files are not followed.
- Roles are tracked per **name**, not per object. A list of DataFrames, a dict, or a value returned from a function loses its role.
- No type inference. `self._vectorizer()` returns "something unknown", so a TF-IDF fit made through it is missed (a real false negative found while dogfooding).
- It does not see data, so it cannot tell whether duplicates or dates actually exist.

## 4. The leakage types, in plain words

Leakage means **information the model would not have at prediction time sneaks into training or evaluation**, so the score is too optimistic.

| Rule | Plain-words version |
|---|---|
| LK001 | You computed the mean/std (scaler), the fill values (imputer), the categories (encoder) or the PCA directions using **all** rows, test rows included, and only then split. |
| LK002 | Your data has time in it, but you split at random. The model learns from the future to predict the past. |
| LK003 | The answer column is still in the inputs. |
| LK004 | You called `.fit` on the test set, so it is no longer unseen. |
| LK005 | You used the test set to decide when to stop training or which hyperparameters to keep. |
| LK006 | SMOTE made synthetic rows from all data, then you split. Test rows have near-twins in train. |
| LK007 | You picked the "best" features by looking at y on all rows. |
| LK008 | You replaced each category with its average target, computed on all rows, so each row's own label is inside its feature. |
| LK009 | You filled missing values with the mean of all rows. |
| LK010 | A rolling average of the target includes today's target. You need `.shift(1)` first. |
| LK011 | You scaled all data once and then ran cross-validation, so every validation fold was seen by the scaler. Use a Pipeline. |
| LK012 | The same patient or user appears many times and you split rows at random, so the model is tested on people it has already seen. |
| LK013 | You scored the test set many times and kept the best, so the test set became a tuning set. |
| LK014 | Group averages or frequency counts were computed over all rows. |
| LK015 | You never removed duplicates, so identical rows can sit in both train and test. |
| LK016 | You glued train and test together and fitted preprocessing on the result (a classic Kaggle habit). |

The general principle: **anything that learns from data (fit, statistics, selection, resampling, encoding) must only see training data**, and the test set is touched once, at the end.

## 5. What I must be able to explain in an interview

1. **The one-sentence pitch:** "A linter for ML data leakage: it reads scripts and notebooks statically, tracks which variables are full, train or test data, and flags fits, splits and evaluations that let test information leak, with a fix hint per rule. It runs in pre-commit and in GitHub code scanning via SARIF."
2. **Why static, not runtime:** zero setup (no data, no GPU, no running a 2-hour notebook), runs in CI in milliseconds, and works on code you can't run. The trade-off: it can't see values, so it misses some leaks and guesses from names.
3. **The core data structure:** the per-name role history `name -> [(line, role)]` and `role_at(name, line)`. Everything else is built on that.
4. **How I keep false positives down:** a confidence level on every finding, name hints only as a fallback, rules that need two pieces of evidence (a fit on full data *and* a later split), good-code fixtures that must stay clean, and dogfooding on real repos. I can describe the three false positives I fixed.
5. **How notebooks work:** code cells are joined in order, magics become `pass`, and a line map turns concatenated line numbers back into (cell, line).
6. **Why SARIF:** it is the standard format GitHub code scanning reads, so findings show up as PR annotations with no server of my own.
7. **How it compares to Yang et al. (ASE '22):** their analysis is deeper (Datalog plus pyright type inference) and covers overlap, multi-test and preprocessing leakage. They found leakage in about 30% of over 100,000 public notebooks. Their tool is a research prototype: it is not on PyPI and needs Soufflé or Docker. Mine is shallower but pip-installable, zero-config and CI-native, and it adds time-series, group, target-encoding, lag-feature and dedup rules. Don't claim to be first: NBLyzer (ICSE-SEIP '22) also did static leakage analysis, and mlguard-lint is another pip-installable linter. See [docs/prior_work.md](docs/prior_work.md).
8. **What I would do next:** follow roles through function calls and containers, add type inference for common return types, and run it on a large public notebook corpus to measure precision in the wild.

## 6. Quiz: re-derive the design (10 problems)

This quiz uses the "rederive" method. Each problem is a design decision the repo made, posed as a problem to solve. Constraints come one at a time, and each one breaks the naive answer. Answer each step in your head (or on paper) **before** opening the answer. The problems follow the order you would build the tool in.

### Problem 1: the input

You want to find leaks in someone's ML code. **How do you get at the code's structure?**

- Constraint A: a regex search for `.fit(` gives hits inside comments and strings.
- Constraint B: calls span several lines, and you need to know which variable was passed in.

<details><summary>Answer</summary>

Parse it into an AST with `ast.parse`. Each call node knows its function, its arguments and its line, so comments and strings are not confused with code. See `source_from_string` / `load_source` in [src/leakray/source.py](src/leakray/source.py).
</details>

### Problem 2: notebooks

Most leaky code lives in `.ipynb` files. **How do you analyse a notebook with the same engine?**

- Constraint A: a notebook is JSON with many cells, and a variable defined in cell 2 is used in cell 5.
- Constraint B: cells contain `%matplotlib inline` and `!pip install`, which are not valid Python.
- Constraint C: the user wants "cell 5, line 2", not "line 47".

<details><summary>Answer</summary>

Join the code cells in order (like "Run all"), replace magic lines with `pass` (same line count), and keep a `cell_map` list from concatenated line to `(cell, line_in_cell)`. A cell that still fails to parse has its bad line blanked and the parse retried. See `notebook_code` and `_parse_lenient` in [src/leakray/source.py](src/leakray/source.py), and `Location.cell` in [src/leakray/models.py](src/leakray/models.py).
</details>

### Problem 3: what a variable "is"

To detect "scaler fitted before the split", you must know whether the data passed to `fit` is full data or training data. **What do you track?**

- Constraint A: `X = df.drop(...)` should still count as full data.
- Constraint B: `X = scaler.fit_transform(X)` reuses the name `X`.
- Constraint C: `both = pd.concat([X_train, X_test])` mixes two kinds.

<details><summary>Answer</summary>

A role per variable name (`full/train/valid/test/unknown`), stored as a history `[(line, role)]` so lookups are "as of" a line (constraint B). New variables combine the roles of their inputs with the ordering `test > valid > train > full` (constraints A and C). See `Role`, `combine`, `role_at` and `role_of` in [src/leakray/dataflow.py](src/leakray/dataflow.py).
</details>

### Problem 4: where the split happens

**How do you know where the train/test boundary is?**

- Constraint A: `train_test_split(X, y)` returns 4 values, and their order matters.
- Constraint B: people also split with `for tr, te in KFold(5).split(X)`.
- Constraint C: time-series people write `train = df[df.date < "2020"]`, with no split function at all.

<details><summary>Answer</summary>

Record `SplitEvent`s. For `train_test_split`, outputs alternate train/test per input. For CV loops, the two loop variables get train/test roles. For slices, a variable whose name says train/test and whose value is derived from full data is a "subset" split, labelled `time_slice` when it compares against a date. See `_assign`, `_handle_for_target` and `_record_subset` in [src/leakray/dataflow.py](src/leakray/dataflow.py).
</details>

### Problem 5: who fitted what

`scaler.fit(X)` and `model.fit(X, y)` look the same in the AST. A scaler fit on full data is a leak, but a final model fit on full data, after evaluation, is fine. **How do you tell them apart?**

- Constraint A: the object was created ten lines earlier: `scaler = StandardScaler()`.
- Constraint B: sometimes it is inline: `StandardScaler().fit_transform(X)`.
- Constraint C: `model = clf.fit(X, y)` makes `model` the same estimator.

<details><summary>Answer</summary>

Remember constructor assignments (`name -> EstimatorDef(cls)`), read the class directly for inline constructors, and alias the result of `.fit()` to the receiver's class. Rules then check the class against knowledge-base sets like `PREPROCESSORS` and `SAMPLERS`. See `FitCall`, `_record_fit` and the sets at the top of [src/leakray/dataflow.py](src/leakray/dataflow.py).
</details>

### Problem 6: one rule, end to end

Write LK001 ("preprocessing fitted before the split") using only the facts above.

- Constraint A: if the file has no split at all (it may be done in another file), you can't be sure.
- Constraint B: if the "later split" is a CV loop, the right advice is "use a Pipeline", not "split first".

<details><summary>Answer</summary>

For each fit whose estimator is a preprocessor and whose X role is `full`, look for a later hold-out (non-CV) split. Report only if one exists (constraint A). CV cases belong to LK011 (constraint B). See [src/leakray/rules/lk001_preprocess_before_split.py](src/leakray/rules/lk001_preprocess_before_split.py) and `later_holdout_split` in [src/leakray/rules/_util.py](src/leakray/rules/_util.py).
</details>

### Problem 7: adding rules without merge conflicts

Sixteen rules, written by different people. **How are they organised so adding one never touches a shared file?**

- Constraint A: the CLI must list all rules (`leakray rules`) and SARIF must describe each one.
- Constraint B: a broken rule must not crash the whole run.

<details><summary>Answer</summary>

One module per rule with an `@register` decorator; modules named `lk*` are auto-discovered with `pkgutil`. Each class carries its own metadata (code, name, severity, confidence, fix). The engine wraps each `check()` in `try/except` and records an error instead of crashing. See [src/leakray/rules/base.py](src/leakray/rules/base.py) and `analyze_source` in [src/leakray/engine.py](src/leakray/engine.py).
</details>

### Problem 8: when the tool is wrong

Users will hit false positives. **What do you give them?**

- Constraint A: they want to silence one line, not a whole rule.
- Constraint B: some rules are guesses (`*_id` columns, "no drop_duplicates") and some are near-certain.
- Constraint C: a team wants different settings per folder.

<details><summary>Answer</summary>

Inline `# leakray: ignore[LK001]` (and `ignore-file`), a confidence value on every finding with `--min-confidence`, and `[tool.leakray]` in pyproject with `per-file-ignores`. See `suppressions` in [src/leakray/engine.py](src/leakray/engine.py) and [src/leakray/config.py](src/leakray/config.py).
</details>

### Problem 9: getting into people's workflow

A linter nobody runs is useless. **How do you get it run on every change?**

- Constraint A: some people want it before each commit.
- Constraint B: teams want results inside the pull request, and they don't want to run a server.
- Constraint C: CI needs a clear pass/fail signal, but info-level findings shouldn't block.

<details><summary>Answer</summary>

`.pre-commit-hooks.yaml` (A); a composite GitHub Action that uploads SARIF to code scanning (B); exit codes 0/1/2 with `--fail-on` and `--exit-zero` (C). See [.pre-commit-hooks.yaml](.pre-commit-hooks.yaml), [action.yml](action.yml), `to_sarif` in [src/leakray/output.py](src/leakray/output.py), and `_cmd_check` in [src/leakray/cli.py](src/leakray/cli.py).
</details>

### Problem 10: proving it works

You claim the tool catches leaks. **How do you show that honestly?**

- Constraint A: unit tests written next to the rules may only test what the rules already do.
- Constraint B: precision matters as much as recall, because a noisy linter gets switched off.
- Constraint C: toy examples are not real code.

<details><summary>Answer</summary>

A separate labelled benchmark written from the rule *definitions*, not the code, with tricky clean cases, reporting per-rule precision and recall ([benchmark/](benchmark/)). Bad/good fixtures per rule as regression tests ([tests/test_fixtures.py](tests/test_fixtures.py)). Dogfooding on real repos, with every finding judged by hand ([docs/dogfood.md](docs/dogfood.md)).
</details>

**Design choices that are approximations, not deep decisions** (say this honestly if asked): treating the whole file as one scope, and the exact list of name-hint tokens (`te`, `va`, `holdout`, ...). These were chosen for speed of building, not proven best.

## 7. Hands-on exercises

### Exercise 1: write a new rule

Write **LK017 `test-size-too-small`**: flag `train_test_split(..., test_size=x)` when `x < 0.05` (a tiny test set gives very noisy scores).

1. Create `src/leakray/rules/lk017_test_size_too_small.py`: copy LK015's structure. Loop over `a.splits`, keep `kind == "train_test_split"`, read `s.call.kwarg_const("test_size")`, and report if it is a float below 0.05.
2. Add `tests/fixtures/LK017/bad_tiny_test.py` and `good_normal_test.py`.
3. Run `pytest tests/test_fixtures.py -k LK017`. `test_every_rule_has_fixtures` will fail until both exist.
4. Add `docs/rules/LK017.md` and the mkdocs nav entry.

### Exercise 2: fix a false negative

The dogfood run found that `vec = self._vectorizer(); X = vec.fit_transform(docs)` is missed because the estimator class is unknown.

1. Write a failing test in `tests/rules/` with that pattern followed by a `train_test_split`.
2. Option A: in `_record_fit`, if the estimator is unknown but the receiver's name contains `vectorizer`, `scaler` or `imputer`, treat it as a preprocessor with **medium** confidence.
3. Make the test pass, run the whole suite, and run `python benchmark/run.py` to check that precision did not drop.

### Exercise 3: measure on real notebooks

1. Clone two permissively licensed public example-notebook repos into a folder **outside** this repo.
2. Run `leakray check <folder> --format json --output out.json`.
3. Open 10 random findings and label each as true positive or false positive. Compute precision from your 10.
4. Write what you learned in a short table, the same way as [docs/dogfood.md](docs/dogfood.md). Do not copy their code into this repo.
