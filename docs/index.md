# leakray

**Find data leakage in ML code before it inflates your results.**

leakray is a fast, zero-config static analyser for Python scripts and Jupyter notebooks. It flags the mistakes that make a model look better in the notebook than it will ever be in production: preprocessing fitted before the split, random splits on time data, the target hiding in the features, tuning on the test set, and 12 more. It parses code with `ast` and never imports or runs it.

```bash
pip install git+https://github.com/tpong-dev/leakray
leakray check .
```

```text
examples\leaky_pipeline.py:18:5 LK001 error, high confidence
  StandardScaler.fit_transform() is fitted on the full dataset before the train/test split on line 24; test-set statistics leak into training.
    18 | X = StandardScaler().fit_transform(X)
  fix: Split first, then fit the transformer on the training data only (scaler.fit(X_train); scaler.transform(X_test)), or put it inside a sklearn Pipeline.
```

## Where to go next

- [Usage](usage.md): CLI, output formats, configuration, suppression.
- [Integrations](integrations.md): pre-commit, GitHub Action and code scanning, notebooks.
- [Rules](rules/index.md): all 16 rules, each with a bad and a good example.
- [Limitations](limitations.md): what a static analyser cannot see.
- [Benchmark](benchmark.md), [Architecture](ARCHITECTURE.md) and [Prior work](prior_work.md).

## Why

Leakage is the most common way ML results go wrong without anyone noticing: the code runs, the score is great, and it does not hold up on new data. Kapoor & Narayanan (2023) found leakage behind reproducibility failures in hundreds of published papers across many fields. Style linters do not look for it. leakray does, and it fits the tools you already use.

Python 3.10 to 3.13, any OS. Its only dependencies are `nbformat` and `rich`. MIT licence.
