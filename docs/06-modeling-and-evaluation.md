# Modeling & evaluation

## Splitting by time, never randomly

`training/baseline.py::temporal_split` splits the dataset by date — `train < train_end`,
`train_end <= val < val_end`, `test >= val_end` — rather than a random `train_test_split`. This
isn't a stylistic preference, it's a correctness requirement for this kind of data.

**Why random splitting leaks information here:** rows in this dataset are correlated both
**spatially** (neighboring cells share weather, and are often part of the same fire event) and
**temporally** (today's weather is strongly correlated with yesterday's — a drought doesn't appear
or vanish between consecutive days). A random split would scatter correlated rows across both train
and test — e.g. cell A on June 5th in train, cell A on June 6th (nearly identical weather) in test.
The model would then get credit in evaluation for essentially having memorized a near-duplicate of a
training row, producing metrics that look good but don't reflect how the model would perform on
genuinely unseen future data. This general failure mode is called **temporal leakage** — see
[glossary.md](glossary.md#leakage).

Splitting by date directly simulates the real deployment scenario: "if this model had existed and
been trained only on data available up to some point, how would it have performed going forward?"
The plan (README, project memory) is train ≤2022, validate on 2023, test on 2024 — each boundary is
a date the model genuinely could not see past during training.

## Baseline-first methodology

Also covered in
[Problem framing](01-problem-framing.md#methodology-baseline-first-complexity-only-if-earned), worth
repeating here with the mechanics: `training/baseline.py` currently implements `fit_dummy()`
(`DummyClassifier`) and `fit_logistic_regression()` (`LogisticRegression`), in that order, before
anything more complex.

**`DummyClassifier(strategy="stratified")`** doesn't look at the features at all — it predicts by
randomly sampling from the *training* label distribution (so with a 0.16% positive rate, it predicts
positive about 0.16% of the time, at random). Its purpose isn't to be a good model; it's a
**floor**. If a real model can't beat this, the pipeline has a bug somewhere upstream (label leakage
the wrong direction, a broken join, features that don't actually vary with the target) — no amount
of model sophistication fixes a broken input, so this check has to happen before investing in a
fancier model.

**`LogisticRegression(class_weight="balanced")`** — the first model that actually uses the features.
`class_weight="balanced"` matters a lot here: by default, a classifier trained on 0.16%-positive
data will often just learn to predict "negative" for everything, since that already gets it 99.84%
training accuracy with near-zero loss contribution from the rare positives.
`class_weight="balanced"` reweights the loss function so misclassifying a rare positive costs
proportionally more than misclassifying a common negative (inversely proportional to class
frequency), which keeps the optimizer from collapsing to "always predict majority class." This is
the standard first-line fix for class imbalance, cheaper than resampling the data and a reasonable
thing to try before resorting to under/oversampling.

## Where `ColumnTransformer` fits

`training/baseline.py::fit_logistic_regression` wires this up as:

```python
FEATURE_COLUMNS = [
    "t2m", "d2m", "u10", "v10", "swvl1", "precip_mm",         # raw weather
    "relative_humidity", "wind_speed", "wind_dir_sin", "wind_dir_cos",
    "days_since_rain", "precip_7d", "precip_30d",              # engineered
    "t2m_mean_7d", "t2m_trend_7d", "rh_mean_7d",
]

model = Pipeline([
    ("scale", StandardScaler()),
    ("logreg", LogisticRegression(class_weight="balanced", max_iter=1000)),
])
model.fit(train[FEATURE_COLUMNS], train[LABEL_COLUMN])
```

A plain `Pipeline`, not a `ColumnTransformer`, in the end — see below for why. A few things worth
being explicit about:

- **No `ColumnTransformer` needed at all, in the end.** Every feature that made it into
`FEATURE_COLUMNS` is numeric (temperature, wind speed, days-since-rain, sin/cos wind direction,
etc.) — there's no categorical column to route separately, so a single `StandardScaler` applied to
everything is enough; `ColumnTransformer` only earns its keep once different columns need different
treatment. `cell_id` is deliberately *not* fed in as a raw feature — treating it as a categorical
column (even one-hot encoded) would let the model partly memorize "this specific cell tends to
burn," which can't generalize to a cell it hasn't seen enough fire history for, and muddies the
temporal-generalization story the whole train/val/test split is designed to test honestly.
- **Scaling only matters for the linear model.** `StandardScaler` is needed for `LogisticRegression`
(gradient-based optimization converges better and regularization behaves sanely when features are on
comparable scales) but is a no-op for tree-based models (`RandomForestClassifier`, `XGBoost`) —
trees split on thresholds per feature independently, so the scale of a feature doesn't change what
splits are chosen. The pipeline can stay a no-scaling passthrough for tree models, or keep the
scaler in place harmlessly.
- **Fit only on train.** Whatever preprocessing goes in the `ColumnTransformer` — scaling, and
eventually imputation for the rolling-feature warm-up NaNs (see
[Feature engineering](05-feature-engineering.md#planned-handling-of-the-nans-this-introduces)) —
gets `.fit()` on the train split only, then `.transform()` on val and test. Fitting on the full
dataset (including val/test) before splitting would leak information about their distribution into
preprocessing decisions, a subtler version of the same leakage problem the temporal split exists to
prevent.

## Metrics

Covered in full in [Problem framing](01-problem-framing.md#the-metrics-used-instead): PR-AUC
(primary), ROC-AUC (secondary context), top-k% capture (operational framing). All three live in
`evaluation/metrics.py` and take `(y_true, y_score)` — raw predicted probabilities/scores, not
thresholded 0/1 predictions, since threshold choice is itself a downstream decision (how much
false-positive tolerance is acceptable given real dispatch costs) that these metrics deliberately
stay agnostic to.

## Baseline results (2023 validation set)

`python -m firesight.training.baseline` splits the real dataset (train ≤2022: 2.59M rows, 4,282
positives; val 2023: 527k rows, 1,135 positives; test 2024: 527k rows, 397 positives — note the
positive count *drops* year over year here, consistent with 2023/2024 being quieter BC fire seasons
than the catastrophic 2021 season sitting in train) and scores both baselines on val:

| model | PR-AUC | ROC-AUC | top-10% capture |
|---|---|---|---|
| `DummyClassifier` | 0.0022 | 0.501 | 5.0% |
| `LogisticRegression` (class-weighted) | 0.0064 | 0.729 | 40.4% |

The dummy floor lands almost exactly at the base rate (~0.0022, matching 2023's ~0.22% positive
rate) and chance ROC-AUC (0.5) — as expected, since it ignores the features entirely. Logistic
regression clears that floor by a wide margin on every metric, most tellingly on the operational
one: **ranking cell-days by this simple linear model and acting on only the riskiest 10% would have
caught 40% of 2023's actual fire detections**, vs. ~10% from acting on a random 10% of cell-days.
That's the first real evidence the whole pipeline — grid, labels, weather join, feature engineering
— captures a genuine signal rather than noise, which is the entire point of running the
dummy/logistic pair before investing in RandomForest/XGBoost (see
[Problem framing](01-problem-framing.md#methodology-baseline-first-complexity-only-if-earned)).
PR-AUC is still low in absolute terms (0.0064) — expected at this base rate; it's the *relative*
jump over the dummy floor that's informative, not the absolute number, which is why the dummy
comparison always ships alongside any single model's score rather than being read in isolation.

## RandomForest and XGBoost (`training/advanced_models.py`)

With the baseline pair confirming the pipeline carries real signal, `advanced_models.py::tune_model`
ran a small manual [grid search](glossary.md#hyperparameter--grid-search) — see its docstring for
why a manual loop over the fixed temporal val split is used instead of
`sklearn.model_selection.GridSearchCV` (its built-in cross-validation would silently reintroduce the
random-split leakage `temporal_split` exists to prevent). Both models get the same class-imbalance
correction in spirit as `LogisticRegression`'s `class_weight="balanced"`: `RandomForestClassifier`
takes the same `class_weight="balanced"` param directly; `XGBClassifier` has no such param, so
`fit_xgboost` computes the equivalent [`scale_pos_weight`](glossary.md#scale_pos_weight)
`= negative_count / positive_count` from the *training fold specifically* each time (recomputed, not
hardcoded, so it can never leak val/test's class balance into the correction).

Best of each grid, by val (2023) PR-AUC — the primary metric — with their 2024 test scores alongside
(test was never used for tuning, purely a final honest read):

| model | params | val PR-AUC | val ROC-AUC | val top-10% | test PR-AUC | test ROC-AUC | test top-10% |
|---|---|---|---|---|---|---|---|
| `LogisticRegression` | class-weighted | 0.0064 | 0.729 | 40.4% | — | — | — |
| `RandomForest` | 400 trees, depth 8, min_leaf 5 | 0.0081 | 0.808 | 43.9% | 0.0054 | 0.813 | 55.4% |
| `XGBoost` | 200 trees, depth 4, lr 0.05 | **0.0083** | 0.805 | **45.1%** | **0.0057** | 0.794 | 49.6% |

Both tree models clear `LogisticRegression` by a solid margin on every val metric, confirming the
extra complexity earns its keep here (see
[Problem framing](01-problem-framing.md#methodology-baseline-first-complexity-only-if-earned) — this
is exactly the check that methodology exists to force). Between the two, it's a genuinely mixed
picture, not a clean win: **XGBoost has the better PR-AUC on both val and test** (the metric this
project treats as primary), but **RandomForest has the better ROC-AUC and top-10% capture on test**
— worth stating plainly rather than picking whichever number favors a predetermined answer.
XGBoost's result was chosen as the currently-served model (`export_model.py::export_current_best`)
on the strength of the primary metric holding up consistently on both the tuning split *and* the
untouched test split, but this is close enough that revisiting it (e.g. a wider hyperparameter grid,
or an ensemble of both) would be a reasonable next step rather than treating XGBoost as a settled
answer.

A general pattern in both grids worth remembering when tuning further: **shallower trees +
more/fewer estimators consistently beat deeper trees** in this search (RandomForest's depth-8
candidates all outscored its depth-16 candidates; XGBoost's depth-4 candidates all outscored its
depth-6 candidates). Consistent with a small positive class (a few thousand positives out of
millions of rows) — deep trees have enough capacity to start fitting noise in the majority class
rather than generalizable fire-weather structure, so shallower trees regularize by construction.
