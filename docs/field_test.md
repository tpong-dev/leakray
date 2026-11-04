# Field test: leakray on public ML example code

Date: 2026-11-04. leakray version 0.1.0 (from the JSON output), run from the repo's `.venv`.

This is a read-only field test. I shallow-cloned four well-known, permissively licensed
repositories of ML example code into a scratch directory outside this repo and ran:

```
leakray check <repo> --format json --exit-zero
```

None of their code was executed and none of it was copied into this repo. The quotes
below are single short lines, there only to identify the flagged statement.

## Repositories

| Repo | URL | Licence (checked in LICENSE file) | Commit (HEAD of shallow clone) | Files analysed | Parse errors |
|---|---|---|---|---|---|
| handson-ml3 | https://github.com/ageron/handson-ml3 | Apache-2.0 | `ef1c9f5ab11083584635b9b3a7846f49c58a6084` | 29 | 1 |
| PythonDataScienceHandbook | https://github.com/jakevdp/PythonDataScienceHandbook | MIT for code (`LICENSE-CODE`); text is CC BY-NC-ND (`LICENSE-TEXT`) | `d66231454ef753818dc9213c9b5942e067266966` | 146 | 1 |
| scikit-learn `examples/` (sparse checkout) | https://github.com/scikit-learn/scikit-learn | BSD-3-Clause | `d5a61d1149cb94a936d4b32212330f4ba8f24ce0` | 282 | 0 |
| ML-For-Beginners | https://github.com/microsoft/ML-For-Beginners | MIT | `de2d4e12236445198213a0711855e348e7253cd4` | 2874 (69 outside `translations/`) | 728 (13 outside `translations/`) |

ML-For-Beginners keeps dozens of translated copies of every lesson under `translations/`.
Those copies produced 770 of its 784 findings and 715 of its 728 parse errors, all of them
duplicates. The numbers below leave `translations/` out, because counting the same notebook
40 times would skew them.

Runtime: 5 to 8 s for each of the first three repos. ML-For-Beginners took 91 s, nearly all
of it in `translations/`.

## Findings per rule

ML-For-Beginners is counted without `translations/`.

| Rule | handson-ml3 | PDSH | sklearn examples | ML-For-Beginners | Total |
|---|---:|---:|---:|---:|---:|
| LK001 preprocessing-fit-before-split | 5 | 4 | 2 | 3 | 14 |
| LK002 random split on time-ordered data | 2 | 0 | 2 | 3 | 7 |
| LK003 target column in features | 0 | 0 | 4 | 0 | 4 |
| LK004 fit on test data | 18 | 2 | 11 | 4 | 35 |
| LK005 tuning / early stopping on test set | 21 | 0 | 1 | 0 | 22 |
| LK007 feature selection before split | 0 | 0 | 1 | 0 | 1 |
| LK011 fit on all rows, then cross-validate | 0 | 0 | 4 | 0 | 4 |
| LK012 random split with repeated entities | 4 | 0 | 0 | 0 | 4 |
| LK013 test score used for selection in a loop | 0 | 0 | 3 | 0 | 3 |
| LK015 no dedup before random split (info) | 8 | 6 | 63 | 4 | 81 |
| **Total** | **58** | **12** | **91** | **14** | **175** |

No rule crashed the tool. Every file that failed to parse was reported as a warning, and the
run carried on.

## Labelled sample

I chose 13 findings by hand, not at random: at least one from each rule family that fired
(LK015 excepted), with extra weight on the two rules that fire most (LK004 and LK005). For
each one I opened the flagged line and the code around it. Verdicts:

- **TP**: a real leak, or real misuse of held-out data.
- **FP**: not a leak.
- **DEBATABLE**: technically a leak with negligible effect, or it hinges on intent.

| # | File : line | Rule | Verdict | Reason |
|---|---|---|---|---|
| 1 | sklearn `examples/calibration/plot_calibration_multiclass.py:72` | LK004 | FP | `clf.fit(X_train, y_train)`: X_train is the training slice `X[:n_train]`, yet the rule reports it as test data. |
| 2 | sklearn `examples/applications/plot_time_series_lagged_features.py:160` | LK004 | FP | Fits on the train fold of a `TimeSeriesSplit` with a 48-row gap, which is the correct method. Reported as fitting on test data. |
| 3 | PDSH `notebooks/05.03-Hyperparameters-and-Model-Validation.ipynb:23` (cell 16) | LK004 | FP | Hand-written two-fold cross-validation: `model.fit(X2, y2).predict(X1)` fits on the second half and scores the first half, on purpose. |
| 4 | ML-For-Beginners `7-TimeSeries/3-SVR/solution/notebook.ipynb:34` (cell 17) | LK004 | FP | Scaler is fitted on the `train` frame only (`scaler.fit_transform(train)`). Reported as test data. |
| 5 | handson-ml3 `10_neural_nets_with_keras.ipynb:248` (cell 69) | LK005 | FP | `validation_data=(X_valid, y_valid)`: X_valid is a validation split carved out of the training set, which is what validation data is meant to be. |
| 6 | handson-ml3 `17_autoencoders_gans_and_diffusion_models.ipynb:746` (cell 147) | LK005 | DEBATABLE | `validation_data=(X_test, X_test)` really does pass the test set. But there are no callbacks and nothing is selected on it, so it is only monitored. |
| 7 | handson-ml3 `13_loading_and_preprocessing_data.ipynb:44` (cell 23) | LK002 | FP | Splits California housing, which has no date column. The `pub_date` column named in the message does not come from this dataset. |
| 8 | sklearn `examples/model_selection/plot_grid_search_digits.py:49` | LK002 | FP | Digits images. The "date-like column" is `mean_score_time`, a `cv_results_` key used later in the file. |
| 9 | sklearn `examples/svm/plot_rbf_parameters.py:129` | LK011 | TP | `StandardScaler` is fitted on all of X, then `GridSearchCV` runs on it. The example's own comment admits it is "cheating a bit". |
| 10 | sklearn `examples/linear_model/plot_tweedie_regression_insurance_claims.py:257` | LK001 | TP | A `ColumnTransformer` with a quantile `KBinsDiscretizer` and `OneHotEncoder` is fitted on the full `df` before `train_test_split` at line 286. |
| 11 | sklearn `examples/applications/plot_digits_denoising.py:41` | LK001 | DEBATABLE | `MinMaxScaler` is fitted before the split. That is a real leak, but on pixel data with a fixed range the effect is close to zero. |
| 12 | sklearn `examples/linear_model/plot_poisson_regression_non_normal_loss.py:147` | LK003 | FP | `df_train` still contains `Frequency`, but the pipeline's `ColumnTransformer` names its feature columns and uses `remainder="drop"`, so the target never reaches the model. |
| 13 | sklearn `examples/neural_networks/plot_mlp_alpha.py:99` | LK013 | FP | `clf.score(X_test, y_test)` inside a loop only writes the score onto a plot. No best model is chosen. The `max`/`min` calls nearby only set axis limits. |

### Sample precision

- TP 2, DEBATABLE 2, FP 9 (n = 13).
- **Strict precision (TP only): 2/13 = 15%.**
- **Lenient precision (TP + DEBATABLE): 4/13 = 31%.**

The sample is small and was chosen by hand, so treat these figures as a rough signal. The
whole-population scan of LK004 and LK005 below points the same way.

### Whole-population check of LK004 and LK005

I read the flagged argument of every LK004 and LK005 finding (outside `translations/`):

- **LK004 (35 findings).** 33 point at a training-like variable: `X_train`, `X_train_A`,
  `X_train_scaled`, `X_cal`, `x_train`, or a frame named `train`. The other 2 are the PDSH
  two-fold cross-validation from row 3. I found **no true positive**. The rule appears to
  treat any output of a split, or anything fitted after a split, as "test data".
- **LK005 (22 findings).** 20 flag `validation_data=(X_valid, ...)` or `validation_data=valid_set`
  in Keras. A validation split is the correct thing to pass there. One is a sklearn
  `search_cv.fit(X_train, y_train)` reported as "tuned on the test set", which is also FP.
  One is row 6 (DEBATABLE). Again, **no clear true positive**.

Together these two rules account for 57 of the 94 non-LK015 findings.

## Parse errors

| Cause | Example | Count (outside `translations/`) |
|---|---|---|
| IPython shell/magic lines inside indented blocks, `%%bash` cells, multi-line `!cmd \` continuations | handson-ml3 `19_training_and_deploying_at_scale.ipynb` (cells 11, 26, 33, 148, 149) | 1 notebook |
| Indented `%timeit` continuation line | PDSH `notebooks/03.12-Performance-Eval-and-Query.ipynb` (cell 5) | 1 notebook |
| R-kernel notebooks (`solution/R/lesson_*-R.ipynb`) | ML-For-Beginners | 9 notebooks |
| 0-byte / empty `.ipynb` files | ML-For-Beginners `2-Regression/1-Tools/notebook.ipynb` | 4 notebooks |

Problems with how these are handled:

1. **One bad cell drops the whole notebook.** The handson-ml3 chapter 19 notebook has more
   than 140 valid Python cells. Five of them use shell magics, and the whole file goes unanalysed.
2. **The error text is not useful.** It reads `could not parse: could not parse <path>`, with
   no cell number, line or reason.
3. **Non-Python kernels are not skipped.** R notebooks (`metadata.kernelspec.language != "python"`)
   are reported as parse errors instead of being skipped quietly.

## Lessons

1. **LK004 is the noisiest rule in the wild.** It is the second most frequent rule (35 hits)
   and showed no true positive. Its logic for telling train variables from test variables is
   wrong on ordinary code: `X_train`, `X_cal`, `train` and the train fold of a CV split are
   all flagged. This should be fixed first, as it does the most harm to user trust.
2. **LK005 confuses validation sets with test sets.** In Keras-heavy code (handson-ml3)
   nearly every `fit()` call gets flagged. It should fire only when the `validation_data`
   variable comes from the test side of a split (for example `X_test`), and possibly only
   when an `EarlyStopping` / checkpoint callback, or later selection, is present.
3. **LK002's date evidence comes from the wrong place.** It picks up column names from
   anywhere in the file or notebook (`pub_date`, `mean_score_time`, the `time` module). The
   evidence should be tied to the DataFrame that is actually being split.
4. **LK003 ignores column selection.** It should not fire when the pipeline's first step is a
   `ColumnTransformer` that lists its columns and uses `remainder="drop"`.
5. **LK013 needs real "select the best" evidence.** Axis-limit `max()`/`min()` calls should not count.
6. **LK001 and LK011 work.** Both true positives in the sample came from these rules. LK001's
   remaining doubtful hits are fit-before-split cases where the effect is negligible (pixel
   MinMax scaling, `LabelEncoder` on a category column, manifold-learning visualisations).
   Severity could be lowered for pure encoders and for unsupervised visualisation.
7. **LK015 is volume noise.** It produced 81 info findings, 63 of them in sklearn's examples,
   most of which use synthetic or built-in datasets with no duplicate risk. Consider turning
   it off by default, or firing only when data is loaded from a file.
8. **Notebook robustness:**
   - Strip or mask IPython magics and shell lines at any indentation, and `%%` cell magics.
   - Skip non-Python kernels.
   - Isolate parse failures per cell instead of per notebook.
   - Report cell and line in parse warnings.
9. **Directory hygiene.** Repos with `translations/` (or other duplicated content) inflate
   counts 50-fold. A default exclude list, or a documented exclude option, would help.
