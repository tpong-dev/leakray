# Dogfooding on real projects

leakray was run **read-only** on three of the author's own ML repositories (nothing in them was changed).
The first run found 4 issues. Each one was checked by hand:

| Repo | Finding | Verdict | What happened |
|---|---|---|---|
| sf311-triage | `train_image.py:228` LK013 (test set scored in a loop) | **False positive, fixed** | The loop scores a *validation* loader that is built with test-time image transforms named `tf_test`. The name hint wrongly made the data "test". Names like `tf_*`, `*_transform` and `*_metric` are no longer treated as data. |
| sf311-triage | `resolution.py:334` LK013 (3 models scored on test, best picked) | **False positive, fixed** | The models are compared on the validation month. The test month only *reports* each model once. The rule now requires the test score to be selected inside or right after a loop. |
| thai-air-forecast | `tests/test_baselines.py:80` LK013 | **False positive, fixed** | A unit test loops over model classes and checks predictions on a fixture called `test`. Nothing is selected, and the old rule matched a `best` variable elsewhere in the file. |
| sf-permit-copilot | `timeline/features.py:320` LK002 (random K-fold on time-ordered data) | **Worth a look (kept)** | Out-of-fold TF-IDF text scores for the training rows are built with `KFold(shuffle=True)` in a project with dated permits. Each training row's text score comes from a model that saw *later* permits. The test months stay held out, so the effect on test scores is probably small, but a time-ordered fold scheme would be cleaner. The same function also fits the TF-IDF vocabulary on all training rows before the folds. leakray did **not** catch that (a false negative), because the vectorizer comes from a helper method, `self._vectorizer()`. |

After the fixes, sf311-triage and thai-air-forecast report no findings, and sf-permit-copilot reports the one LK002 above.

**Takeaway:** all three repos already used forward time splits and held-out validation sets, which is the lesson leakray encodes. The false positives came from name heuristics, which is the known weak spot of a name- and AST-based analysis. Each one became a regression test case.
