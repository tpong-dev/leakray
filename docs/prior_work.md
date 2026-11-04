# Prior work

leakray is not the first tool or paper to look for data leakage in ML code. This page lists the closest prior work, what each one does, and where leakray sits relative to it. Facts were checked against the primary sources linked under [References](#references) in November 2026. Anything that could not be checked is marked *(not verified)*.

## Comparison

| Tool / paper | Approach | Static or runtime | Install | CI / SARIF | Maintained | Licence |
|---|---|---|---|---|---|---|
| **Yang et al., ASE 2022** ([leakage-analysis](https://github.com/malusamayo/leakage-analysis)) | Datalog queries over a pointer and dataflow analysis, with type inference from a customised pyright | Static | From source (Souffle + pyright submodule + `requirements.txt`, Python 3.8 recommended) or Docker. Not on PyPI. Notebooks must be converted to `.py` first | No CI integration or SARIF output documented | Research artefact. Last push 2 May 2023 | MIT |
| **NBLyzer** (Subotić et al., ICSE-SEIP 2022) | Abstract interpretation over notebook cells, aware of out-of-order cell execution. Includes a data leakage analysis | Static | Described as a notebook server and VS Code extension. No public release found *(not verified)* | Not described | Unknown *(not verified)* | Unknown *(not verified)* |
| **LeakageDetector** (AlOmar et al., arXiv 2025) | PyCharm plugin that wraps Yang et al.'s analysis (run in Docker) and adds quick-fix skeletons | Static | PyCharm plugin, from [GitHub](https://github.com/SE4AIResearch/DataLeakage_Fall2023) | No | Unknown *(not verified)* | Unknown *(not verified)* |
| **mlguard-lint** (PyPI) | Heuristic `ast`-based linter for scikit-learn workflows, 21 rules (MLG001 to MLG021), several of them leakage rules | Static | `pip install mlguard-lint` | CI gate via exit code; text/JSON output; SARIF and pre-commit not mentioned | One release line, latest 0.1.1 (June 2026) | MIT |
| **leaklint** (PyPI) | Checks the actual train/test data and pipelines for leakage | Runtime (data) | `pip install leaklint` | CLI with non-zero exit, SARIF output, sample pre-commit hook (per PyPI page) | One release, 0.1.0 (June 2026) | MIT |
| **deepchecks** (train-test validation suite) | Checks on the data, e.g. train/test sample mix, date leakage (duplicates/overlap), index leakage, feature-label correlation change | Runtime (data and model) | `pip install deepchecks` | Python API and reports. No SARIF | Latest PyPI release 0.19.1 (Dec 2024) | AGPL-3.0 |
| **Kapoor & Narayanan, Patterns 2023** | Survey of prior reviews, taxonomy of 8 leakage types, "model info sheets" checklist | Not a tool (manual checklist) | n/a | n/a | n/a | Open access article |
| **Kaufman et al., TKDD 2012** | Formal definition of leakage, "learn-predict separation", ways to detect leakage in data | Not a tool (methodology) | n/a | n/a | n/a | n/a |
| **leakray** | Python `ast` plus a lightweight name/role dataflow. No type inference, no Datalog, never runs code | Static | `pip install leakray`, zero config | Text/JSON/SARIF, pre-commit hook, GitHub Action to code scanning | Active | MIT |

## Each item

### Yang, Brower-Sinning, Lewis & Kästner (ASE 2022)

"Data Leakage in Notebooks: Static Detection and Better Processes" was published at ASE '22, the 37th IEEE/ACM International Conference on Automated Software Engineering (DOI [10.1145/3551349.3556918](https://doi.org/10.1145/3551349.3556918); preprint [arXiv:2209.03345](https://arxiv.org/abs/2209.03345)). It defines three leak types:

- **Overlap leakage**: test rows used directly in training or tuning, or training rows derived from test rows through augmentation or oversampling.
- **Multi-test leakage**: the same test data is evaluated repeatedly and used to make decisions, so it is no longer unseen.
- **Preprocessing leakage**: train and test data are transformed together (scaling, feature selection, PCA, text vectorisation), so test data shapes the training transformation.

The method is a static dataflow analysis built on a 2-call-site-sensitive, Andersen-style pointer analysis, implemented as Datalog facts and queries, with pyright for type inference. On 100 hand-labelled notebooks it was 92.9% accurate (262 of 282 potential leaks). Run on more than 100,000 public notebooks, it found some leakage in 29.6% of the GitHub notebooks: preprocessing 12.3%, overlap 6.5%, multi-test 18.5%. The paper also proposes process changes to prevent leakage. The authors suggest the analyser could run in notebook environments or code review, but the paper describes no IDE or Jupyter extension. The repository links to an online demo *(not opened)*.

### NBLyzer (Subotić, Milikić & Stojić, ICSE-SEIP 2022)

"A Static Analysis Framework for Data Science Notebooks" (ICSE-SEIP '22, DOI [10.1145/3510457.3513032](https://doi.org/10.1145/3510457.3513032); preprint [arXiv:2110.08339](https://arxiv.org/abs/2110.08339)) comes from Microsoft. NBLyzer uses abstract interpretation on individual cells and models out-of-order cell execution, so it can warn that running a given cell can cause leakage further down. Among its analyses the paper presents "a novel data leakage analysis", which predates Yang et al. by a few months. It was evaluated on 2,211 real notebooks, and 98.7% of them were analysed in under a second. The authors say it was built as a notebook server and a VS Code extension. I found no public release.

### LeakageDetector (AlOmar et al., 2025)

"LeakageDetector: An Open Source Data Leakage Analysis Tool in Machine Learning Pipelines" ([arXiv:2503.14723](https://arxiv.org/abs/2503.14723)) is a PyCharm plugin. It runs Yang et al.'s analyser in Docker, shows overlap, multi-test and preprocessing leaks in the IDE, and offers quick fixes that the authors call a skeleton for a real fix. It brings Yang et al.'s analysis into an IDE, but it has the same three leak types and the same heavy setup.

### mlguard-lint (PyPI)

The closest tool in form. It is a pip-installable static linter that parses `.py` and `.ipynb` with `ast`, does not execute code, and has 21 rules for scikit-learn workflows. These include transformers fitted before the split, preprocessing outside CV, group leakage, target encoding without cross-fitting, target column in features, ID columns as features, and duplicates before the split. It can gate CI through its exit code. Its README does not mention SARIF or pre-commit. It calls itself heuristic. It first appeared on PyPI in June 2026, so how well it is maintained is not yet known.

### leaklint (PyPI)

A runtime tool that inspects actual datasets and train/test splits instead of source code. Its PyPI page describes SARIF output and a sample pre-commit hook, and it presents itself as complementary to static analysis. First released in June 2026.

### deepchecks

A widely used validation library (AGPL-3.0). Its tabular train-test validation checks include `train_test_samples_mix`, `date_train_test_leakage_duplicates`, `date_train_test_leakage_overlap`, `index_leakage` and `feature_label_correlation_change`. These checks run on real data and models after the code has run, so they catch leaks that static analysis cannot see, such as duplicates in raw files. They cannot point to the line of code that caused a leak.

### Kapoor & Narayanan (Patterns, 2023)

"Leakage and the reproducibility crisis in machine-learning-based science", *Patterns* 4(9), 100804 (DOI [10.1016/j.patter.2023.100804](https://doi.org/10.1016/j.patter.2023.100804); preprint [arXiv:2207.07048](https://arxiv.org/abs/2207.07048)). The authors surveyed earlier reviews across fields that use ML and found 17 fields where leakage had been reported, affecting at least 294 papers. They give a taxonomy of eight leakage types and propose "model info sheets", and a case study shows complex ML models losing their edge over logistic regression for civil war prediction once leakage is fixed. The published numbers differ from the 2022 preprint, which says 329 papers. The authors' project page now lists a larger, updated count (648 papers across 30 fields), which is not the figure in the published article.

### Kaufman, Rosset, Perlich & Stitelman (TKDD, 2012)

"Leakage in Data Mining: Formulation, Detection, and Avoidance", *ACM Transactions on Knowledge Discovery from Data* 6(4), article 15 (DOI [10.1145/2382577.2382579](https://doi.org/10.1145/2382577.2382579)). It extends a KDD 2011 paper by the first three authors (DOI 10.1145/2020408.2020496). It defines leakage as information about the target that would not legitimately be available at prediction time, and proposes "learn-predict separation" plus detection methods for data. Yang et al. cite it as mainly about finding label leakage manually in raw data.

## How leakray differs

Weaker:

- **Shallower analysis.** leakray tracks names and roles (train, test, full data, target) through assignments in one file. It has no pointer analysis, no type inference and no Datalog. Yang et al.'s analysis follows values through aliasing, containers, function calls and library semantics, which leakray does not.
- **More false negatives.** Leaks hidden behind aliases, lists or dicts of frames, helper functions, or splits done in another module are often missed. A split done elsewhere is only recognised through names such as `train_df` or `test_df`.
- **No published accuracy figure yet** to compare with Yang et al.'s 92.9%, and no measurement of how often it is wrong across a large corpus.
- **No runtime view.** Like any static tool, it cannot see duplicates or overlaps that only exist in the data. deepchecks or leaklint cover that.

Stronger:

- **Install and run.** `pip install leakray`, no config, no Datalog engine, no Docker, no type checker. It is fast enough to run on every commit.
- **Scripts and notebooks.** It reads `.py` and `.ipynb` directly, with no conversion step.
- **CI-native.** Text, JSON and SARIF output, a pre-commit hook, and a GitHub Action that uploads findings to GitHub code scanning.
- **Broader rule set.** 16 rules. Besides Yang et al.'s three categories (preprocessing before split, fit on test, eval_set or tuning on test, oversampling before split, feature selection before split, repeated test evaluation, concatenating train and test), it covers random splits on time series, target in features, target encoding on full data, global `fillna`, unshifted target lags, preprocessing outside a CV pipeline, KFold ignoring groups, groupby-transform before split and missing deduplication.
- **A fix hint for every rule**, in the output.
- **A public labelled benchmark** in `benchmark/`, so false positive and false negative rates can be checked by anyone.

Compared with mlguard-lint, which is closest in form, leakray adds SARIF and code scanning, a pre-commit hook and a GitHub Action. The two tools overlap in many rules, and nobody has compared their accuracy on a shared benchmark.

## What leakray borrows

- **Yang et al.'s three categories.** Overlap, multi-test and preprocessing leakage are the backbone of the rule set: fit on test, eval_set on test and oversampling before split are overlap; repeated test evaluation is multi-test; preprocessing, feature selection and fillna before the split are preprocessing leakage. Their finding that these patterns are common in real notebooks is the main reason to check for them by default.
- **Kapoor & Narayanan's taxonomy.** The rules beyond Yang et al.'s categories come from their eight types, especially features that are illegitimate proxies for the target (target in features, unshifted lags), temporal leakage (random splits on time series) and non-independence between train and test (group-aware CV, deduplication, groupby before split).
- **Kaufman et al.'s definition** of leakage as information that would not be available at prediction time, which is how leakray decides whether a pattern counts as leakage.
- **Fix suggestions**, as in LeakageDetector's quick fixes, though leakray's are text hints, not code edits.

## References

Pages opened while writing this document:

1. Yang, C., Brower-Sinning, R. A., Lewis, G. A., & Kästner, C. (2022). Data Leakage in Notebooks: Static Detection and Better Processes. *ASE '22*. https://doi.org/10.1145/3551349.3556918. Preprint: https://arxiv.org/abs/2209.03345, full text: https://ar5iv.labs.arxiv.org/html/2209.03345, record: https://par.nsf.gov/biblio/10444835
2. leakage-analysis repository: https://github.com/malusamayo/leakage-analysis (activity and licence from the GitHub API, https://api.github.com/repos/malusamayo/leakage-analysis)
3. Subotić, P., Milikić, L., & Stojić, M. (2022). A Static Analysis Framework for Data Science Notebooks. *ICSE-SEIP '22*. https://doi.org/10.1145/3510457.3513032. PDF: https://www.microsoft.com/en-us/research/uploads/prod/2022/05/ICSE.pdf, page: https://www.microsoft.com/en-us/research/publication/a-static-analysis-framework-for-data-science-notebooks/, preprint: https://arxiv.org/abs/2110.08339
4. AlOmar, E. A., DeMario, C., Shagawat, R., & Kreiser, B. (2025). LeakageDetector: An Open Source Data Leakage Analysis Tool in Machine Learning Pipelines. https://arxiv.org/abs/2503.14723 (HTML: https://arxiv.org/html/2503.14723v1)
5. Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns*, 4(9), 100804. https://doi.org/10.1016/j.patter.2023.100804. PDF via https://par.nsf.gov/servlets/purl/10513990, record: https://par.nsf.gov/biblio/10513990, project page: https://reproducible.cs.princeton.edu/, preprint: https://arxiv.org/abs/2207.07048
6. Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in Data Mining: Formulation, Detection, and Avoidance. *ACM TKDD*, 6(4), article 15. https://doi.org/10.1145/2382577.2382579 (DOI and metadata confirmed from search results only, ACM page not opened)
7. mlguard-lint on PyPI: https://pypi.org/pypi/mlguard-lint/json
8. leaklint on PyPI: https://pypi.org/pypi/leaklint/json
9. deepchecks: https://github.com/deepchecks/deepchecks, train-test checks: https://github.com/deepchecks/deepchecks/tree/main/deepchecks/tabular/checks/train_test_validation, PyPI: https://pypi.org/pypi/deepchecks/json
