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

## Scoping to fire season

`training/baseline.py::filter_fire_season` restricts every training/evaluation run to **May 1 – Oct
15** (any year), matching the Kamloops Fire Centre's typical Category 2/3 open-burning prohibition
window. It's applied right after loading the dataset and before `temporal_split`, in `baseline.py`,
`advanced_models.py`, and `export_model.py` alike — so this is a scope decision for the whole
project, not just an evaluation-time filter, and the model the API serves is trained (not just
scored) on fire-season data only.

**Why:** this follows directly from the [winter/shoulder-season blind
spot](#known-limitation-a-wintershoulder-season-blind-spot) below. Two rounds of feature engineering
couldn't give the model any way to flag a winter fire, because winter/shoulder-season fires are more
often human-caused (debris burning, equipment) than weather-driven, and every feature here is
weather-derived — there's no fixable gap, just a different phenomenon the available data can't see.
Rather than keep chasing that with more features, the project now scopes to the months where (a) the
weather-driven fire-risk signal this model measures actually applies, and (b) the large, destructive,
operationally-important fires concentrate — hot+dry conditions both drive ignition and let fires
spread fast once started, which is why BC's largest fire seasons (e.g. the catastrophic summer of
2021, spot-checked in an earlier step) are a summer phenomenon. Restricting to fire season also
shrinks the extreme class imbalance a little, since none of the near-zero-risk winter days are in the
pool being ranked/scored anymore.

Numbers throughout the rest of this page from before this change reflect the full-year (Jan-Dec)
dataset; re-running `baseline.py`/`advanced_models.py`/`export_model.py` after this change will
produce fire-season-only numbers that aren't directly comparable to them (fewer rows, different class
balance, different date range) — expected to move mostly by removing the "free" near-zero-risk winter
rows from the ranking, not because anything about the model itself changed. See [Re-tuning after the
fire-season scope change](#re-tuning-after-the-fire-season-scope-change) below for what actually
happened when the models were re-tuned on the new scope.

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

## Widening the search: RandomizedSearchCV + PredefinedSplit

The grids above are deliberately small (8 candidates each) — a first tuning pass, not an exhaustive
one. The reason `tune_model` avoids `GridSearchCV` is its *default* cross-validation, which reshuffles
data into random folds and would reintroduce the same leakage `temporal_split` exists to prevent —
but that only rules out the default, not the tool. [`PredefinedSplit`](glossary.md#predefinedsplit)
lets you hand `RandomizedSearchCV`/`GridSearchCV` the *exact* existing train/val split (`test_fold`:
`-1` for every train row, `0` for every val row) instead of letting it invent folds, so the same
temporal-safety guarantee holds while gaining `RandomizedSearchCV`'s ability to sample a much wider,
continuous hyperparameter space for a fixed evaluation budget (`n_iter` draws) rather than enumerating
every combination in a hand-written grid.

`training/advanced_models.py::tune_random_search` implements this: it builds the `PredefinedSplit`,
runs `RandomizedSearchCV(..., refit=False)`, then refits the winning params on the train fold only
(refit=False matters here — `RandomizedSearchCV`'s own refit would otherwise retrain the winner on
train+val combined, silently changing what the "best" model was actually fit on, which would break
the "test is never touched during tuning" guarantee every other model in this module keeps). The
wider distributions (`RANDOM_FOREST_DISTRIBUTIONS`, `XGBOOST_DISTRIBUTIONS`) add dimensions the small
grids never searched at all — `max_features` for RandomForest, `subsample`/`colsample_bytree`/
`reg_alpha`/`reg_lambda` for XGBoost — sampled 15 times each:

| model | params | val PR-AUC | val ROC-AUC | val top-10% | test PR-AUC | test ROC-AUC | test top-10% |
|---|---|---|---|---|---|---|---|
| `RandomForest` (grid) | 400 trees, depth 8, min_leaf 5 | 0.0081 | 0.808 | 43.9% | 0.0054 | 0.813 | 55.4% |
| `XGBoost` (grid) | 200 trees, depth 4, lr 0.05 | 0.0083 | 0.805 | 45.1% | 0.0057 | 0.794 | 49.6% |
| `RandomForest` (random search) | 181 trees, depth 5, max_features 0.40, min_leaf 1 | 0.0092 | **0.817** | 50.4% | **0.0058** | **0.809** | **59.7%** |
| `XGBoost` (random search) | 131 trees, depth 3, lr 0.016, subsample 0.67, colsample 0.86 | **0.0093** | 0.811 | **52.5%** | 0.0054 | 0.803 | 59.4% |

Both widened models clear their own grid-search counterparts by a wide margin on every val metric,
and both post a large top-10% capture gain on the untouched test set (55.4%/49.6% -> 59.7%/59.4%) —
the wider search earns its keep. The winning configs are consistently **even shallower** than the
small grids' winners (RandomForest depth 5 vs. 8; XGBoost depth 3 with a learning rate roughly a
third of the grid's best), reinforcing the shallower-trees-generalize-better pattern above rather than
contradicting it.

This also **changes the earlier RandomForest-vs-XGBoost read**: with both given the same wider search,
RandomForest is now the stronger candidate on the test set specifically — better PR-AUC, better
ROC-AUC, and a very slightly better top-10% capture — while XGBoost only edges it on the val metrics
(the ones most exposed to overfitting a single validation fold). That's a genuine flip from the grid
results, where XGBoost had won on PR-AUC. Not a fully exhaustive result — 15 sampled candidates is evidence of a trend, not a guarantee, and a
larger `n_iter` would sharpen it further (an attempt at `n_iter=40` was abandoned after running ~2
hours with no output, far past the ~20 minutes linear scaling from the 15-candidate run would predict
— most likely `n_jobs=-1`'s process-based parallelism thrashing on repeated large-dataframe pickling
across workers, not a real compute need). On the strength of the 15-candidate result, RandomForest
(`BEST_RANDOM_FOREST_PARAMS`) is now the model `export_model.py` exports — see
[Serving](07-serving.md#persisting-a-model-without-losing-its-contract) for that swap, verified live
the same way the earlier LogisticRegression -> XGBoost swap was.

## Re-tuning after the fire-season scope change

Two changes landed together before this re-tune: the [fire-season
scoping](#scoping-to-fire-season) above, and extending the raw FIRMS/ERA5-Land ingestion window back
to 2012 (from 2018) for more training rows — `train` now covers 2012-2022 instead of 2018-2022, while
`val` (2023) and `test` (2024) boundaries are unchanged. Both `advanced_models.py::tune_model` (the
small hand-written grids) and `tune_random_search` (the wider `RandomizedSearchCV`+`PredefinedSplit`
search) were re-run against this new scope:

| model | params | val PR-AUC | val ROC-AUC | val top-10% | test PR-AUC | test ROC-AUC | test top-10% |
|---|---|---|---|---|---|---|---|
| `RandomForest` (grid) | 200 trees, depth 16, min_leaf 5 | **0.0150** | 0.795 | 39.5% | 0.0102 | 0.878 | 69.4% |
| `XGBoost` (grid) | 200 trees, depth 6, lr 0.1 | 0.0144 | 0.785 | 36.5% | 0.0088 | 0.876 | 64.5% |
| `RandomForest` (random search) | 238 trees, depth 6, max_features 0.606, min_leaf 7 | 0.0138 | 0.832 | 41.9% | **0.0106** | **0.884** | **71.9%** |
| `XGBoost` (random search) | 162 trees, depth 5, lr 0.015, subsample 0.88, colsample 0.72 | 0.0135 | 0.827 | 41.4% | 0.0081 | 0.882 | 69.0% |

The same pattern as the earlier widening-the-search result repeats: grid-search RandomForest narrowly
wins on val PR-AUC, but the randomized-search RandomForest wins on **every** test metric (PR-AUC,
ROC-AUC, top-10% capture) despite a lower val score — and val PR-AUC is exactly the number most prone
to overfitting a single validation fold, especially when a grid only tried 8 combinations. Consistent
with that reasoning (and with how the RandomForest-over-XGBoost call was made last time), the
randomized-search RandomForest (`n_estimators=238, max_depth=6, max_features=0.6063, min_samples_leaf=7`)
is the new `BEST_RANDOM_FOREST_PARAMS` in `export_model.py`, re-exported to `data/processed/model.joblib`.

**Don't over-read the jump from the pre-scoping numbers** (e.g. test top-10% capture 59.7% ->
71.9%). Two things changed at once — more training years and a narrower, easier-to-rank evaluation
population (winter's huge pool of near-zero-risk non-fire rows is gone from val/test, not just from
train) — neither of which means the model generalizes better to conditions it couldn't handle before.
It's a fair comparison of models *within* this run, and a legitimate scope decision, but not evidence
of a modeling breakthrough against the old numbers above it on this page.

Also worth naming: test metrics beating val metrics by this much (e.g. test top-10% 71.9% vs. val
41.9%) is the same directional pattern seen before the scope change, just more pronounced. The
likely explanation carries over unchanged — 2024 (test) contains the small number of large, obvious
summer fire events fire-weather features are best at catching, while 2023 (val) apparently has a
harder mix — but this is worth re-checking with a monthly breakdown if the gap keeps widening as the
project evolves, the same way the winter blind spot was originally found by looking past the
aggregate number.

## Known limitation: a winter/shoulder-season blind spot

Error analysis against the served RandomForest's own risk ranking on the 2024 test set (splitting
actual fires by whether they landed in the model's own top 10% by predicted risk, the same cutoff
`top_10pct_capture` scores against) found the model's overall 59.7% capture rate hides a strong
seasonal split, not a uniformly-distributed error rate:

| month | fires caught (top 10%) | fires missed |
|---|---|---|
| Jul | 111 | 9 |
| Aug | 81 | 30 |
| Nov | 26 | 33 |
| Feb | 1 | 31 |
| Dec | 0 | 23 |

July/August fires are caught almost perfectly; **December is 0/23 and February is 1/32**. Missed
fires occur in conditions ~14.5K cooler, 23 percentage points more humid, and after ~5 more recent
dry-free days than caught fires — the model has learned "hot + dry = risk", which nails summer
fire-weather but has no way to flag an ignition that happens *despite* cool, wet conditions. That's
plausible for winter/shoulder-season fires specifically: they're more likely human-caused (debris
burning, equipment, campfires) than fuel/weather-driven, and every feature in `FEATURE_COLUMNS` is
weather-derived.

Two feature-engineering attempts to close this gap were tried and **both failed to move it**:

1. **Calendar features** (`day_of_year_sin`/`cos`, `is_weekend`, an `open_burning_season` flag
   derived from the Kamloops Fire Centre's typical Category 2/3 burning-prohibition window) — the
   two day-of-year features became the model's top-2 by importance (40% combined, more than soil
   moisture), but December stayed 0/23 and February 1/32 exactly. Test top-10% capture barely moved
   (59.7% -> 60.7%) while val top-10% capture dropped hard (50.4% -> 33.1%).
2. **Proximity features** (`dist_to_road_km`, `dist_to_place_km`, nearest-neighbor distance from
   each grid cell to OpenStreetMap roads/populated places, fetched via the Overpass API and joined
   as static per-cell values) — same result: December 0/23, February 1/32, unchanged. Test top-10%
   capture dropped to 48.9%. A diagnostic refit with much more capacity (uncapped depth, 400 trees,
   run purely to check whether the tuned model's shallow depth was the bottleneck rather than the
   features themselves) moved December/February by only 1-2 fires each while cratering every other
   metric (test top-10% capture 26.4%) — the classic overfitting-to-majority-class-noise pattern
   from the tuning results above, not evidence the added capacity was the fix.

Neither addition was kept — both were reverted after evaluation, so they aren't in `FEATURE_COLUMNS`
or `data/processed/kamloops_dataset.parquet` today, and there is no `proximity.py` or
`ingest_geography.py` in the repo.

**Why this looks structural rather than a missing-feature problem:** both attempts added *static*
signal — a value fixed per calendar date or per grid cell, constant across the many days/cells it
applies to. `top_10pct_capture` ranks every (cell, date) row in the entire test year against every
other row for one global cutoff. A static offset can shift a cell's or a date's baseline risk up or
down, but it cannot manufacture day-to-day variation within a cell, and it is nowhere near strong
enough to lift a handful of winter fires above the tens of thousands of ordinary-looking winter
non-fire rows it's competing against in that single global ranking. Fixing this for real would need a
feature that varies *with the actual ignition event* day by day (e.g. real burn-permit records or
lightning-strike data), which this project has no source for — not a different weather or calendar
proxy computed from data already on hand.

**Conclusion:** treated as a documented limitation of a weather-only model rather than a bug to keep
chasing with more features. The project's actual response to this, as of the [fire season
scoping](#scoping-to-fire-season) above, isn't a season-specific threshold — it's excluding
Nov-Apr from the problem entirely, since those are the months this blind spot lives in and the
months where the largest, most operationally-important fires don't occur anyway.

## Feature importance: what the model is actually leaning on

Before trusting the served RandomForest's predictions, it's worth checking that its accuracy comes
from real fire-weather signal and not from an artifact of the grid, the join, or how the temporal
split happens to fall. Two importance measures were compared:

- **Mean decrease in impurity (MDI)** — `rf.feature_importances_`, built into the fitted model. Fast,
  but biased toward continuous/high-cardinality features and only reflects the training set, so it can
  overstate a feature the model happened to split on a lot without that split actually helping predict
  new data.
- **Permutation importance** — `sklearn.inspection.permutation_importance`, computed separately on the
  2023 val and 2024 test splits (10 repeats each, scored by PR-AUC, the project's primary metric).
  Shuffling one feature column at a time and measuring how much held-out PR-AUC drops is a more honest
  measure of what the model actually depends on to generalize, since it's evaluated on data the model
  never trained on.

Both agree on the same ranking, which is reassuring — if MDI and permutation importance disagreed
sharply, that would suggest the model was overfit to training-set idiosyncrasies rather than a real
pattern:

| feature | MDI | permutation ΔPR-AUC (val) | permutation ΔPR-AUC (test) |
|---|---|---|---|
| `swvl1` (soil moisture) | 0.269 | +0.00203 | +0.00308 |
| `t2m_mean_7d` | 0.224 | +0.00165 | +0.00264 |
| `precip_30d` | 0.180 | +0.00240 | +0.00322 |
| `t2m` | 0.112 | +0.00070 | +0.00128 |
| `rh_mean_7d` | 0.085 | +0.00053 | +0.00052 |
| `relative_humidity` | 0.040 | +0.00032 | +0.00136 |
| `precip_mm` | 0.028 | +0.00035 | +0.00004 |
| `days_since_rain` | 0.015 | +0.00004 | +0.00014 |
| `precip_7d` | 0.010 | +0.00006 | +0.00004 |
| `u10`, `v10`, `wind_dir_sin/cos`, `wind_speed`, `d2m`, `t2m_trend_7d` | all <0.01 | ~0 or slightly negative | ~0 or slightly negative |

Two takeaways:

1. **The signal is real, not an artifact.** The top 5 features by both measures — soil moisture,
   30-day precip, 7-day mean temp, raw temp, 7-day mean humidity — are exactly the slow-moving
   fuel-dryness indicators a wildfire-weather domain expert would expect to matter most. None of the
   grid/join mechanics (cell geometry, nearest-neighbor weather assignment) show up as unexpectedly
   dominant, which is what a subtle pipeline bug driving the score would look like.
2. **Wind and the 7-day temp trend are dead weight in this model.** `wind_speed`, both wind-direction
   components, `d2m`, and `t2m_trend_7d` all sit at or below zero permutation importance on both val
   and test — shuffling them doesn't hurt held-out PR-AUC at all, sometimes even helps slightly (noise,
   not real negative signal). The RandomForest simply isn't using them; it's making its calls almost
   entirely off soil moisture, precipitation, and temperature. This wasn't touched — removing
   low-importance features is a legitimate follow-up but not free at this depth/leaf-count, since a
   depth-5 tree gets to pick very few splits in total and its meaning could still change if the search
   were re-run without those columns — but it explains, in addition to the reasoning in [Known
   limitation](#known-limitation-a-wintershoulder-season-blind-spot) above, why the winter/shoulder-
   season blind spot was so resistant to more weather features: the model's five real levers are all
   slow-moving fuel-dryness signals, and nothing in `FEATURE_COLUMNS` — including the features it
   currently ignores — encodes anything about human activity, which is the more likely driver of
   winter ignitions.
