# Limitations

leakray is a static analyser: it reads code and never runs it. That makes it fast and safe, but there are things it cannot see.

## What it cannot see

- **Other files.** It reads one file at a time. A split done in another module, or data that was already split on disk, is only recognised through names like `train_df` / `test_df`.
- **Data values.** Duplicates, actual timestamps, near-duplicate images, or a feature that is a proxy of the target need the data itself.
- **Indirect flows.** Roles are tracked per variable name and are not followed through lists, dicts or function returns.
- **Custom code.** Estimators and helpers are recognised by class or function name. Hand-written transformers or samplers with unusual names are missed.

## Name hints

Some rules use hints (`*_id`, `date`, `target`). They report medium or low confidence, so use `--min-confidence` to tune noise. A column named `user_id` that is unique per row is not a group; a column named `date` may not make the rows a time series.

## False positives and negatives

Each [rule page](rules/index.md) has a "When it can be wrong" section for that rule. Passing leakray does not prove a pipeline is leak-free; it only means the patterns it knows were not found. Leakage can also come from the way data was collected, which no code check can find.

False positives are bugs. [Report them](https://github.com/tpong-dev/leakray/issues/new?template=false_positive.yml).
