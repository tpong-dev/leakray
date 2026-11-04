# Rules

leakray has 16 rules. Each finding has a severity (`error`, `warning`, `info`) and a confidence (high, medium, low). Use `--min-confidence` and `--fail-on` to tune them, and `leakray explain LKxxx` for the short version of any page below.

| Code | Name | Severity | Confidence | Catches |
|---|---|---|---|---|
| [LK001](LK001.md) | preprocessing-fit-before-split | error | high | scaler / imputer / encoder / PCA fitted on all rows before the split |
| [LK002](LK002.md) | random-split-on-time-series | warning | high | random split or shuffled K-fold on dated data |
| [LK003](LK003.md) | target-in-features | error | high | the target column is still in `X` |
| [LK004](LK004.md) | fit-on-test-data | error | high | `.fit` / `.fit_transform` on the test set |
| [LK005](LK005.md) | tuning-on-test-set | error | high | `eval_set` / `validation_data` / search / objective on the test set |
| [LK006](LK006.md) | oversampling-before-split | error | high | SMOTE / over- or under-sampling before the split or CV |
| [LK007](LK007.md) | feature-selection-before-split | error | high | supervised feature selection on all rows |
| [LK008](LK008.md) | target-encoding-on-full-data | error | high | target / mean encoding on all rows |
| [LK009](LK009.md) | global-fillna-before-split | warning | medium | `fillna(df.mean())` before the split |
| [LK010](LK010.md) | unshifted-target-lag-feature | error | high | rolling / lag of the target without `.shift()` |
| [LK011](LK011.md) | preprocessing-outside-cv-pipeline | error | high | preprocessing fitted on all rows, then cross-validation |
| [LK012](LK012.md) | kfold-ignores-groups | warning | medium | random K-fold when an id column calls for groups |
| [LK013](LK013.md) | repeated-test-evaluation | warning | medium | scoring the test set in a loop and keeping the best |
| [LK014](LK014.md) | groupby-transform-before-split | warning | medium | group aggregates / frequency encoding over all rows |
| [LK015](LK015.md) | no-dedup-before-random-split | info | low | random split with no `drop_duplicates()` |
| [LK016](LK016.md) | concat-train-test-before-fit | error | high | `pd.concat([train, test])` then fitting preprocessing |

## Grouped by where the leak happens

- **Preprocessing fitted on too much data:** LK001, LK009, LK011, LK014, LK016
- **Labels reaching the features:** LK003, LK007, LK008, LK010
- **Test set used for decisions:** LK004, LK005, LK013
- **Splits that do not match the data:** LK002, LK006, LK012, LK015
