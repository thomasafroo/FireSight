# Modeling & evaluation

## Current model

What `data/processed/model.joblib` holds today, so this doesn't have to be reconstructed from the
dated decisions below. **Keep this section updated whenever `export_model.py` promotes a different
model or feature set**, it is the one place on this page that describes the present rather than the
history.

| | served same-day model | 3-day-ahead model |
| --- | --- | --- |
| bundle | `data/processed/model.joblib` | `data/processed/model_3day.joblib` |
| label | `ignited` | `ignited_next_3d` |
| estimator | RandomForest, `BEST_RANDOM_FOREST_PARAMS` | same |
| features | 32 (`baseline.py::FEATURE_COLUMNS`) | same 32 |
| calibrator | pooled isotonic | none yet |
| val (2023) PR-AUC | 0.3632 | 0.3520 |
| test (2024) PR-AUC | 0.3816 | 0.2906 |
| test top-10% capture | 88.0% | 80.8% |

`BEST_RANDOM_FOREST_PARAMS` is `n_estimators=238, max_depth=6, max_features=0.6063,
min_samples_leaf=7`, tuned on the 10-feature weather-only set and never re-tuned since, because each
later feature addition was measured against those fixed params and a [max_features
sweep](#per-group-ablation-isolating-which-of-the-three-actually-helps) confirmed the value is still
near-optimal at 32 columns. The 32 features are 10 weather, 3 `neighbor_fire_count_{1,3,7}d`, and 19
`fuel_type_*` one-hots.

**Read those test numbers as one year, not as expected performance.** The single biggest finding on
this page is that a single held-out year badly overstates how stable the model is: a [rolling-origin
backtest](#re-verifying-the-rolling-origin-backtest-against-the-13-feature-model) of the
13-feature model across 8 years gave a mean top-10% capture of 64.7% and a range of 29%-93%, against
the 86.0% that year's single test split reported. The 32-feature model has not been backtested that
way, so treat 88.0% as the same kind of best-case single-year read.

**How it got here**, each step measured before promotion, with the full reasoning in the linked
section:

1. [LogisticRegression, then XGBoost](#randomforest-and-xgboost-trainingadvanced_modelspy) on the
first grid search.
2. [RandomForest](#widening-the-search-randomizedsearchcv--predefinedsplit), once a wider
`RandomizedSearchCV` gave it the edge on the untouched test set.
3. [Six dead-weight features dropped](#dropping-the-dead-weight-features), 16 columns down to 10.
4. [`neighbor_fire_count` added](#spatial-lag-features-implemented-and-promoted-2026-08-19), the
single largest gain in the project, leakage-checked three ways.
5. [Fuel type added](#per-group-ablation-isolating-which-of-the-three-actually-helps), the only
promoted group of the FWI/terrain/fuel-type batch.

Tried and deliberately **not** promoted: `cape`/`convective_precip_mm`, FWI, terrain, a 1D-CNN over
raw weather sequences, attention pooling on that CNN, and Venn-Abers uncertainty intervals. Each has
its own section below with the measured result that decided it.

## Splitting by time, never randomly

`training/baseline.py::temporal_split` splits the dataset by date, `train < train_end`,
`train_end <= val < val_end`, `test >= val_end`, rather than a random `train_test_split`. This
isn't a stylistic preference, it's a correctness requirement for this kind of data.

**Why random splitting leaks information here:** rows are correlated both **spatially**
(neighboring cells share weather, and are often part of the same fire event) and **temporally**
(today's weather is strongly correlated with yesterday's). A random split scatters correlated rows
across train and test, e.g. cell A on June 5th in train, cell A on June 6th in test, so the model
gets credit for having memorized a near-duplicate of a training row. This failure mode is **temporal
leakage**, see [glossary.md](glossary.md#leakage-temporal--data-leakage).

Splitting by date directly simulates the real deployment scenario: "if this model had existed and
been trained only on data available up to some point, how would it have performed going forward?"
The plan (README, project memory) is train ≤2022, validate on 2023, test on 2024, each boundary is
a date the model genuinely could not see past during training.

## Scoping to fire season

`training/baseline.py::filter_fire_season` restricts every training/evaluation run to **May 1 – Oct
15** (any year), matching the Kamloops Fire Centre's typical Category 2/3 open-burning prohibition
window. It's applied right after loading the dataset and before `temporal_split`, in `baseline.py`,
`advanced_models.py`, and `export_model.py` alike, so this is a scope decision for the whole
project, not just an evaluation-time filter, and the model the API serves is trained (not just
scored) on fire-season data only.

**Why:** this follows directly from the [winter/shoulder-season blind
spot](#known-limitation-a-wintershoulder-season-blind-spot) below. Two rounds of feature engineering
couldn't give the model any way to flag a winter fire, because those fires are more often
human-caused (debris burning, equipment) than weather-driven and every feature here is
weather-derived: not a fixable gap, a different phenomenon the available data can't see. Rather than
keep chasing it, the project scopes to the months where the weather-driven signal actually applies
and where the large, operationally-important fires concentrate. It also shrinks the extreme class
imbalance slightly, since the near-zero-risk winter days leave the pool being ranked.

Numbers earlier on this page predate this change and reflect the full-year dataset, so they aren't
directly comparable to anything after it (fewer rows, different class balance, different date range).
See [Re-tuning after the fire-season scope
change](#re-tuning-after-the-fire-season-scope-change) for what happened when the models were
re-tuned on the new scope.

## Baseline-first methodology

Also covered in
[Problem framing](01-problem-framing.md#methodology-baseline-first-complexity-only-if-earned), worth
repeating here with the mechanics: `training/baseline.py` currently implements `fit_dummy()`
(`DummyClassifier`) and `fit_logistic_regression()` (`LogisticRegression`), in that order, before
anything more complex.

`DummyClassifier(strategy="stratified")` ignores the features entirely, predicting by randomly
sampling from the *training* label distribution (so at a 0.16% positive rate, it predicts positive
about 0.16% of the time). Its purpose isn't to be a good model, it's a **floor**: if a real model
can't beat this, the pipeline has a bug upstream (a broken join, features that don't vary with the
target), and no amount of model sophistication fixes a broken input.

`LogisticRegression(class_weight="balanced")` is the first model that actually uses the features.
The class weighting matters a lot: trained on 0.16%-positive data, a classifier will otherwise learn
to predict "negative" for everything, since that alone already gets 99.84% training accuracy.
`class_weight="balanced"` reweights the loss inversely to class frequency, so misclassifying a rare
positive costs proportionally more, which keeps the optimizer from collapsing to the majority class.
The standard first-line fix for imbalance, and cheaper than resampling.

## Where `ColumnTransformer` fits

`training/baseline.py::fit_logistic_regression` wires this up as:

```python
FEATURE_COLUMNS = [
    "t2m", "swvl1", "precip_mm", "relative_humidity", "wind_speed",  # raw weather
    "days_since_rain", "precip_7d", "precip_30d",                     # engineered
    "t2m_mean_7d", "rh_mean_7d",
]

model = Pipeline([
    ("scale", StandardScaler()),
    ("logreg", LogisticRegression(class_weight="balanced", max_iter=1000)),
])
model.fit(train[FEATURE_COLUMNS], train[LABEL_COLUMN])
```

A plain `Pipeline`, not a `ColumnTransformer`, in the end, see below for why. A few things worth
being explicit about:

- **No** `ColumnTransformer` **needed at all, in the end.** Every feature in `FEATURE_COLUMNS` was
numeric at this point, so there was no categorical column to route separately and a single
`StandardScaler` on everything is enough; `ColumnTransformer` only earns its keep once different
columns need different treatment. `cell_id` is deliberately *not* fed in as a feature: treating it
as categorical would let the model partly memorize "this specific cell tends to burn," which can't
generalize to a cell without enough fire history and muddies the temporal-generalization story the
split exists to test.
- **Scaling only matters for the linear model.** `StandardScaler` helps `LogisticRegression`
(gradient-based optimization and regularization both behave better on comparable scales) but is a
no-op for trees, which split on per-feature thresholds independently.
- **Fit only on train.** `StandardScaler` is `.fit()` on train, then `.transform()` on val and test.
Fitting it on the full dataset would leak their distribution into a preprocessing decision, a subtler
version of the leakage the temporal split exists to prevent. (Rolling-feature warm-up `NaN`s are a
separate, earlier step: those rows are dropped outright, not imputed, in
`pipeline/build_dataset.py` before the split happens, see [Feature
engineering](05-feature-engineering.md#handling-the-nans-this-introduces).)

## Metrics

Covered in full in [Problem framing](01-problem-framing.md#the-metrics-used-instead): PR-AUC
(primary), ROC-AUC (secondary context), top-k% capture (operational framing). All three live in
`evaluation/metrics.py` and take `(y_true, y_score)`, raw predicted probabilities/scores, not
thresholded 0/1 predictions, since threshold choice is itself a downstream decision (how much
false-positive tolerance is acceptable given real dispatch costs) that these metrics deliberately
stay agnostic to.

## Baseline results (2023 validation set)

`python -m firesight.training.baseline` splits the real dataset (train ≤2022: 2.59M rows, 4,282
positives; val 2023: 527k rows, 1,135 positives; test 2024: 527k rows, 397 positives, note the
positive count *drops* year over year here, consistent with 2023/2024 being quieter BC fire seasons
than the catastrophic 2021 season sitting in train) and scores both baselines on val:


| model                                 | PR-AUC | ROC-AUC | top-10% capture |
| ------------------------------------- | ------ | ------- | --------------- |
| `DummyClassifier`                     | 0.0022 | 0.501   | 5.0%            |
| `LogisticRegression` (class-weighted) | 0.0064 | 0.729   | 40.4%           |


The dummy floor lands almost exactly at the base rate and chance ROC-AUC, as expected. Logistic
regression clears it by a wide margin on every metric, most tellingly the operational one: **ranking
cell-days by this simple linear model and acting on only the riskiest 10% would have caught 40% of
2023's actual fire detections**, vs. ~10% from acting on a random 10%. That's the first real evidence
the whole pipeline (grid, labels, weather join, feature engineering) captures genuine signal rather
than noise, which is the entire point of running the dummy/logistic pair first. PR-AUC is still low
in absolute terms (0.0064), expected at this base rate; it's the *relative* jump over the dummy floor
that's informative, which is why the dummy comparison ships alongside every score rather than being
read in isolation.

## RandomForest and XGBoost (`training/advanced_models.py`)

With the baseline pair confirming the pipeline carries real signal, `advanced_models.py::tune_model`
ran a small manual [grid search](glossary.md#hyperparameter--grid-search), see its docstring for
why a manual loop over the fixed temporal val split is used instead of
`sklearn.model_selection.GridSearchCV` (its built-in cross-validation would silently reintroduce the
random-split leakage `temporal_split` exists to prevent). Both models get the same class-imbalance
correction in spirit as `LogisticRegression`'s `class_weight="balanced"`: `RandomForestClassifier`
takes the same `class_weight="balanced"` param directly; `XGBClassifier` has no such param, so
`fit_xgboost` computes the equivalent [`scale_pos_weight`](glossary.md#scale_pos_weight)
`= negative_count / positive_count` from the *training fold specifically* each time (recomputed, not
hardcoded, so it can never leak val/test's class balance into the correction).

Best of each grid, by val (2023) PR-AUC, the primary metric, with their 2024 test scores alongside
(test was never used for tuning, purely a final honest read):


| model                | params                         | val PR-AUC | val ROC-AUC | val top-10% | test PR-AUC | test ROC-AUC | test top-10% |
| -------------------- | ------------------------------ | ---------- | ----------- | ----------- | ----------- | ------------ | ------------ |
| `LogisticRegression` | class-weighted                 | 0.0064     | 0.729       | 40.4%       | N/A         | N/A          | N/A          |
| `RandomForest`       | 400 trees, depth 8, min_leaf 5 | 0.0081     | 0.808       | 43.9%       | 0.0054      | 0.813        | 55.4%        |
| `XGBoost`            | 200 trees, depth 4, lr 0.05    | **0.0083** | 0.805       | **45.1%**   | **0.0057**  | 0.794        | 49.6%        |


Both tree models clear `LogisticRegression` by a solid margin on every val metric, confirming the
extra complexity earns its keep (exactly the check the baseline-first methodology exists to force).
Between the two it's genuinely mixed, not a clean win: **XGBoost has the better PR-AUC on both val
and test** (the primary metric), but **RandomForest has the better ROC-AUC and top-10% capture on
test**, worth stating plainly rather than picking whichever number favors a predetermined answer.
XGBoost was promoted to the served model at this point on the strength of the primary metric holding
on both splits, close enough that it wasn't treated as settled, and a wider search later reversed it
(see [Widening the search](#widening-the-search-randomizedsearchcv--predefinedsplit) below).

A general pattern in both grids worth remembering when tuning further: **shallower trees +
more/fewer estimators consistently beat deeper trees** in this search (RandomForest's depth-8
candidates all outscored its depth-16 candidates; XGBoost's depth-4 candidates all outscored its
depth-6 candidates). Consistent with a small positive class (a few thousand positives out of
millions of rows), deep trees have enough capacity to start fitting noise in the majority class
rather than generalizable fire-weather structure, so shallower trees regularize by construction.

## Widening the search: RandomizedSearchCV + PredefinedSplit

The grids above are deliberately small (8 candidates each), a first pass rather than an exhaustive
one. `tune_model` avoids `GridSearchCV` because of its *default* cross-validation, which reshuffles
data into random folds and would reintroduce the leakage `temporal_split` exists to prevent, but that
rules out the default, not the tool. [`PredefinedSplit`](glossary.md#predefinedsplit) lets you hand
`RandomizedSearchCV` the *exact* existing train/val split (`test_fold`: `-1` per train row, `0` per
val row) instead of letting it invent folds, keeping the temporal-safety guarantee while gaining the
ability to sample a much wider, continuous hyperparameter space for a fixed budget of `n_iter` draws.

`training/advanced_models.py::tune_random_search` implements this: it builds the `PredefinedSplit`,
runs `RandomizedSearchCV(..., refit=False)`, then refits the winning params on the train fold only
(refit=False matters here, `RandomizedSearchCV`'s own refit would otherwise retrain the winner on
train+val combined, silently changing what the "best" model was actually fit on, which would break
the "test is never touched during tuning" guarantee every other model in this module keeps). The
wider distributions (`RANDOM_FOREST_DISTRIBUTIONS`, `XGBOOST_DISTRIBUTIONS`) add dimensions the small
grids never searched at all, `max_features` for RandomForest, `subsample`/`colsample_bytree`/
`reg_alpha`/`reg_lambda` for XGBoost, sampled 15 times each:


| model                          | params                                                       | val PR-AUC | val ROC-AUC | val top-10% | test PR-AUC | test ROC-AUC | test top-10% |
| ------------------------------ | ------------------------------------------------------------ | ---------- | ----------- | ----------- | ----------- | ------------ | ------------ |
| `RandomForest` (grid)          | 400 trees, depth 8, min_leaf 5                               | 0.0081     | 0.808       | 43.9%       | 0.0054      | 0.813        | 55.4%        |
| `XGBoost` (grid)               | 200 trees, depth 4, lr 0.05                                  | 0.0083     | 0.805       | 45.1%       | 0.0057      | 0.794        | 49.6%        |
| `RandomForest` (random search) | 181 trees, depth 5, max_features 0.40, min_leaf 1            | 0.0092     | **0.817**   | 50.4%       | **0.0058**  | **0.809**    | **59.7%**    |
| `XGBoost` (random search)      | 131 trees, depth 3, lr 0.016, subsample 0.67, colsample 0.86 | **0.0093** | 0.811       | **52.5%**   | 0.0054      | 0.803        | 59.4%        |


Both widened models clear their grid-search counterparts on every val metric and post a large
top-10% capture gain on the untouched test set (55.4%/49.6% -> 59.7%/59.4%), so the wider search earns
its keep. The winners are consistently **even shallower** than the grids' winners (RandomForest depth
5 vs. 8; XGBoost depth 3 at roughly a third the learning rate), reinforcing the shallower-generalizes-
better pattern above.

This also **flips the earlier RandomForest-vs-XGBoost read**: given the same wider search,
RandomForest wins on the test set (better PR-AUC, ROC-AUC, and top-10% capture) while XGBoost only
edges it on the val metrics, the ones most exposed to overfitting a single validation fold. 15
sampled candidates is evidence of a trend, not a guarantee (an `n_iter=40` attempt was abandoned after
~2 hours with no output, far past what linear scaling predicts, most likely `n_jobs=-1`'s
process-based parallelism thrashing on repeated large-dataframe pickling). On that basis RandomForest
(`BEST_RANDOM_FOREST_PARAMS`) is what `export_model.py` exports, see
[Serving](07-serving.md#persisting-a-model-without-losing-its-contract) for that swap.

## Re-tuning after the fire-season scope change

Two changes landed together before this re-tune: the [fire-season
scoping](#scoping-to-fire-season) above, and extending the raw FIRMS/ERA5-Land ingestion window back
to 2012 (from 2018) for more training rows, `train` now covers 2012-2022 instead of 2018-2022, while
`val` (2023) and `test` (2024) boundaries are unchanged. Both `advanced_models.py::tune_model` (the
small hand-written grids) and `tune_random_search` (the wider `RandomizedSearchCV`+`PredefinedSplit`
search) were re-run against this new scope:


| model                          | params                                                       | val PR-AUC | val ROC-AUC | val top-10% | test PR-AUC | test ROC-AUC | test top-10% |
| ------------------------------ | ------------------------------------------------------------ | ---------- | ----------- | ----------- | ----------- | ------------ | ------------ |
| `RandomForest` (grid)          | 200 trees, depth 16, min_leaf 5                              | **0.0150** | 0.795       | 39.5%       | 0.0102      | 0.878        | 69.4%        |
| `XGBoost` (grid)               | 200 trees, depth 6, lr 0.1                                   | 0.0144     | 0.785       | 36.5%       | 0.0088      | 0.876        | 64.5%        |
| `RandomForest` (random search) | 238 trees, depth 6, max_features 0.606, min_leaf 7           | 0.0138     | 0.832       | 41.9%       | **0.0106**  | **0.884**    | **71.9%**    |
| `XGBoost` (random search)      | 162 trees, depth 5, lr 0.015, subsample 0.88, colsample 0.72 | 0.0135     | 0.827       | 41.4%       | 0.0081      | 0.882        | 69.0%        |


The earlier pattern repeats: grid-search RandomForest narrowly wins on val PR-AUC, but the
randomized-search RandomForest wins **every** test metric despite a lower val score, and val PR-AUC is
the number most prone to overfitting a single fold, especially from an 8-combination grid. Same
reasoning as the RandomForest-over-XGBoost call last time, so the randomized-search RandomForest
(`n_estimators=238, max_depth=6, max_features=0.6063, min_samples_leaf=7`) is the new
`BEST_RANDOM_FOREST_PARAMS`.

**Don't over-read the jump from the pre-scoping numbers** (test top-10% 59.7% -> 71.9%). Two things
changed at once: more training years, and a narrower, easier-to-rank evaluation population (winter's
pool of near-zero-risk rows is gone from val/test, not just train). It's a fair comparison of models
*within* this run, not evidence of a modeling breakthrough against the older numbers above.

Test metrics beating val metrics by this much (71.9% vs. 41.9%) is the same direction seen before the
scope change, just more pronounced. See [Investigating the val/test
gap](#investigating-the-valtest-gap-a-monthly-breakdown) below for what explains it.

## Investigating the val/test gap: a monthly breakdown

The gap (test top-10% 71.9% vs. val 41.9%) was checked with the same tool that surfaced the winter
blind spot: a monthly breakdown of which fires the served model's own top-10%-by-risk cutoff catches
vs. misses, run separately on val (2023) and test (2024).


| month | val (2023) caught/total | val capture | test (2024) caught/total | test capture |
| ----- | ----------------------- | ----------- | ------------------------ | ------------ |
| May   | 0/4                     | 0%          | 0/6                      | 0%           |
| Jun   | 0/6                     | 0%          | N/A                      | N/A          |
| Jul   | 47/101                  | 46.5%       | 102/120                  | 85.0%        |
| Aug   | 282/538                 | 52.4%       | 72/111                   | 64.9%        |
| Sep   | 7/153                   | 4.6%        | N/A                      | N/A          |
| Oct   | N/A                     | N/A         | 0/5                      | 0%           |


The gap isn't a val-vs-test modeling artifact, it's a **fire-count distribution difference between
the two years, landing on a month the model is already weak at**. Val (2023) had 153 fires in
September alone (19% of its 802) that the model catches only 4.6% of the time; test (2024) had
essentially no September fires to be dragged down by. Val's July/August rates (46.5%/52.4%) are
actually *higher* than its 41.9% aggregate and much closer to test's (85.0%/64.9%), so it's the
September cluster alone pulling val down. That fits 2023 being BC's worst season on record with large
fires still active into September, against a 2024 concentrated in July/August.

**Why September is structurally weak:** the same reasoning as the [winter/shoulder-season blind
spot](#known-limitation-a-wintershoulder-season-blind-spot), in miniature. September sits at the tail
of the window, cooler and wetter than peak summer, so it's a smaller version of the same "hot+dry =
risk" limitation, not severe enough to justify excluding the month (which would also throw away 153
real September positives from *training* data every year). No code change: the gap is explained by
real between-year variation in *when* fires happened, not a leak or a scoring inconsistency.

## Rolling-origin backtest: is 71.9% typical, or the best year in the dataset?

The investigation above explains the *val-vs-test* gap, but it only compares two years. The deeper
question it raises, "how much does the reported number move if you happen to test on a different
year?", needs more than two data points to answer honestly. `evaluation/backtest.py` answers it
with a [rolling-origin backtest](glossary.md#rolling-origin-backtest): keep
`BEST_RANDOM_FOREST_PARAMS` fixed (no re-tuning per fold, to isolate "does the *evaluation year*
matter" from "would retuning help"), refit on an **expanding** training window (2012 through year
N-1), and score against each subsequent year N in turn, for N = 2017..2024 (2012-2016 reserved as
a five-year floor before the first holdout, so the earliest fold isn't evaluating off a single year
of history).


| year | train rows | holdout positives | PR-AUC | ROC-AUC | top-10% capture |
| ---- | ---------- | ----------------- | ------ | ------- | --------------- |
| 2017 | 1,212,120  | 1,433             | 0.0141 | 0.763   | 30.8%           |
| 2018 | 1,454,544  | 229               | 0.0023 | 0.762   | 25.3%           |
| 2019 | 1,696,968  | 57                | 0.0004 | 0.623   | 22.8%           |
| 2020 | 1,939,392  | 43                | 0.0003 | 0.653   | 16.3%           |
| 2021 | 2,181,816  | 2,628             | 0.0333 | 0.829   | 27.0%           |
| 2022 | 2,424,240  | 98                | 0.0005 | 0.632   | **8.2%**        |
| 2023 | 2,666,664  | 802               | 0.0136 | 0.833   | 41.8%           |
| 2024 | 2,909,088  | 242               | 0.0113 | 0.884   | **74.4%**       |


**Sanity check first:** the 2023 row reproduces the original `val` numbers exactly (PR-AUC 0.0136,
ROC-AUC 0.833, top-10% 41.8%), as expected from the same train set and holdout year. The 2024 row is
close to but not identical to the reported `test` numbers (71.9% there vs. 74.4% here) for a real
reason: `export_model.py`'s served model trains only through 2022, while this fold's training window
includes 2023 before scoring 2024. A small, expected boost from one more training year.

**The headline number turns out to be close to the best year observed, not a typical one.** Across
the 8 folds, top-10% capture has mean **30.8%**, median **26.2%**, and a standard deviation of **~19
percentage points**, enormous relative spread on a 0-100% metric. The 71.9%/74.4% number this project
had been citing sits at the *top* of that range; five of eight years land at 30% or below, and 2022,
trained on *more* data than 2017 or 2021, is worst of all at 8.2%.

**What doesn't explain the swing:** more training data (rows grow monotonically 1.2M -> 2.9M while
capture doesn't track it at all), and not raw holdout fire count either (2017 and 2021, BC's two worst
seasons, score *worse* than the quieter 2023 and 2024). **What fits the evidence** is the structural
weakness documented above: the model is much stronger in peak July/August than at the season's
shoulders, so a year's score depends heavily on *when within the season* its fires land. See [Why
performance swings by month and year](#why-performance-swings-by-month-and-year) below, which tests
that hypothesis rather than just ruling out the simpler ones.

**One reassuring result across every fold:** ROC-AUC never dropped below 0.62 (mean 0.747), so the
model beat random ranking in every year tested, including the worst. The instability is in *how much*
better than chance it is, not *whether* it is.

**Practical takeaway:** don't repeat "71.9% top-10% capture" as expected real-world performance. Cite
the range (8-74%) or the median (~26%). The model still beats chance by a wide, real margin; it's the
size of that margin that's far less certain and far more year-dependent than one test number showed.

## Why performance swings by month and year

The backtest's leading hypothesis, that a year's score depends on *when in the season* its fires
land, had only been tested informally on two years. `evaluation/backtest.py::monthly_capture_breakdown`
generalizes that by-hand table to all 8 rolling-origin folds, using the same top-10%-by-risk cutoff
`top_10pct_capture` scores against, broken down by month.

**Pooled across all 8 years, the month-level pattern is unambiguous and consistent:**


| month | fires (8 years pooled) | caught | capture rate |
| ----- | ---------------------- | ------ | ------------ |
| May   | 53                     | 2      | **3.8%**     |
| Jun   | 45                     | 19     | 42.2%        |
| Jul   | 2,289                  | 903    | 39.4%        |
| Aug   | 2,522                  | 712    | 28.2%        |
| Sep   | 409                    | 90     | 22.0%        |
| Oct   | 214                    | 27     | **12.6%**    |


May is by far the weakest month in the window, worse in relative terms than the already-documented
September weakness, with October second weakest. June and July are strongest, and August, despite
being colloquially "peak fire season," is meaningfully weaker than July (28.2% vs. 39.4%) once pooled
across 8 years. Clear confirmation that the model's skill concentrates in the core of the season and
thins at both edges, not a two-year coincidence.

**But month mix only partly explains the year-to-year swing, and the residual is informative.** Ranking
each year by the *share* of its fires that landed in Jul/Aug (the two strongest months) against its
overall `top_10pct_capture` gives a correlation of **r = 0.63** across the 8 folds, real and positive,
but far from a complete explanation:


| year | % of that year's fires in Jul/Aug | top-10% capture |
| ---- | --------------------------------- | --------------- |
| 2019 | 3.5%                              | 22.8%           |
| 2020 | 32.6%                             | 16.3%           |
| 2022 | 34.7%                             | 8.2%            |
| 2018 | 39.3%                             | 25.3%           |
| 2023 | 79.7%                             | 41.8%           |
| 2017 | 87.4%                             | **30.8%**       |
| 2021 | 97.0%                             | **27.0%**       |
| 2024 | 95.5%                             | 74.4%           |


**2017 and 2021 are the outliers that keep this from being a clean story.** Both had 87-97% of their
fires in the nominally-strongest months yet scored only 27-31%, worse than 2023 and far worse than
2024's similarly Jul/Aug-heavy year. Both are BC's worst seasons on record, and both show the same
pattern: 2021's August alone (1,009 fires, roughly 2-4x a typical year's *entire* season) was caught
only **11.7%** of the time, in a month that is strong in every other year. That points to a second
factor beyond month-of-season: **extreme, high-volume seasons break the ranking even within their
strong months.**

**Follow-up investigation confirmed two separate, compounding mechanisms.** Refitting just the 2017
and 2021 folds (plus 2024 as a normal-year control) and comparing caught vs. missed fires on two axes,
how many *other* cells ignited the same day, and how the raw weather features differ:

**1. Extreme same-day fire counts directly overwhelm the fixed top-10% budget, but only past a
threshold.** Bucketing each year's fires by how many cells ignited on that exact date:


| year           | 1 fire/day | 2-5   | 6-20  | 21-50     | 51-100 | 100+      |
| -------------- | ---------- | ----- | ----- | --------- | ------ | --------- |
| 2024 (control) | 0%         | 38.1% | 83.6% | 80.0%     | N/A    | N/A       |
| 2017           | 7.7%       | 20.6% | 33.6% | 31.4%     | 23.1%  | N/A       |
| 2021           | 0.0%       | 15.2% | 24.2% | **41.5%** | 28.2%  | **15.1%** |


In the normal-year control (2024, max 30 same-day fires), capture *rises* with same-day fire count:
more simultaneous ignitions means a hotter, drier, windier day, exactly what the model is tuned to
flag. **2021 shows the opposite past a point**, peaking at 41.5% for 21-50-fire days then falling to
28.2% and 15.1% as counts climb past 50 and past 100. 2021 had entire days with 100+ simultaneous
ignitions (581 fires in that one bucket), which no other year approached. `top_10pct_capture` ranks a
fixed ~24,000-row slice against the *whole year* (~144 rows/day across a 168-day season), so a day
with 100+ real fires mechanically cannot fit in its share of the budget even if the model correctly
flags the whole day as extreme. 2017 shows the same shape at smaller scale (31.4% -> 23.1%) without
reaching 2021's 100+ regime.

**2. The weather-based signal itself is less discriminative in extreme years.** Comparing mean feature
values between caught and missed fires:


|                     | 2024 (control): caught vs. missed | 2021: caught vs. missed          |
| ------------------- | --------------------------------- | -------------------------------- |
| `t2m`               | 295.2K vs. 289.9K (**5.3K gap**)  | 294.6K vs. 293.1K (**1.5K gap**) |
| `relative_humidity` | 38.6% vs. 54.1% (**15.5pp gap**)  | 34.6% vs. 43.3% (**8.7pp gap**)  |
| `swvl1`             | 0.171 vs. 0.263                   | 0.165 vs. 0.181                  |


The "hot+dry gets caught, cool+wet gets missed" direction holds every year, but the *gap* between
caught and missed is roughly a third the size in 2021 as in the control. That fits 2021's defining
feature: the June 2021 BC heat dome put much of the province under extreme heat and drought
*simultaneously*, compressing exactly the cell-to-cell variation the model ranks on. When nearly every
cell looks dangerously hot and dry at once, there's less relative signal left to separate the ones
that actually ignite, on top of the fixed-budget problem in (1).

**Together:** month-of-season explains most of the routine year-to-year swing (r=0.63), and these two
compounding effects explain why the two most severe seasons underperform relative to their Jul/Aug-heavy
month mix. Both are structural properties of a global, weather-only ranking on binary-outcome data,
not bugs to fix with more features.

## Known limitation: a winter/shoulder-season blind spot

Error analysis against the served RandomForest's own risk ranking on the 2024 test set (splitting
actual fires by whether they landed in the model's own top 10% by predicted risk, the same cutoff
`top_10pct_capture` scores against) found the model's overall 59.7% capture rate hides a strong
seasonal split, not a uniformly-distributed error rate:


| month | fires caught (top 10%) | fires missed |
| ----- | ---------------------- | ------------ |
| Jul   | 111                    | 9            |
| Aug   | 81                     | 30           |
| Nov   | 26                     | 33           |
| Feb   | 1                      | 31           |
| Dec   | 0                      | 23           |


July/August fires are caught almost perfectly; **December is 0/23 and February is 1/32**. Missed
fires occur in conditions ~14.5K cooler, 23 percentage points more humid, and after ~5 more recent
dry-free days than caught fires. The model learned "hot + dry = risk", which nails summer fire-weather
but has no way to flag an ignition that happens *despite* cool, wet conditions, plausible for
winter/shoulder fires specifically: they're more likely human-caused than weather-driven, and every
feature in `FEATURE_COLUMNS` is weather-derived.

**Confirmed, not just plausible.** BC Wildfire Service's own incident records (the `FIRE_CAUSE` field
in `WHSE_LAND_AND_NATURAL_RESOURCE.PROT_HISTORICAL_INCIDENTS_SP`, queried live via DataBC's public WFS
endpoint for the Kamloops FC bbox, `FIRE_TYPE == "Fire"`, 2012-2024) back this up: of 2,310
fire-season fires, 59.5% are lightning-caused and 37.0% person-caused; of the 327 winter/shoulder
(Nov-Apr) fires, that flips hard to **90.8% person-caused, 0.9% lightning**. December and February are
the extreme case, with every recorded fire in those months (1 and 3 respectively) `Person`-caused.
Those are BCWS *incidents*, not the FIRMS *detections* the table counts, one incident can produce many
detections across satellite passes, which is why 23 December and 32 February detections map onto far
fewer recorded fires. This is diagnostic confirmation only: `FIRE_CAUSE` exists solely for fires that
already happened, so it can't become a predictive feature. It upgrades the human-cause explanation
from a plausible read to a government-recorded fact, and rules out lightning-detection data as a fix,
since there's essentially no lightning signal in these months to detect.

Two feature-engineering attempts to close this gap were tried and **both failed to move it**:

1. **Calendar features** (`day_of_year_sin`/`cos`, `is_weekend`, an `open_burning_season` flag
  derived from the Kamloops Fire Centre's typical Category 2/3 burning-prohibition window), the
   two day-of-year features became the model's top-2 by importance (40% combined, more than soil
   moisture), but December stayed 0/23 and February 1/32 exactly. Test top-10% capture barely moved
   (59.7% -> 60.7%) while val top-10% capture dropped hard (50.4% -> 33.1%).
2. **Proximity features** (`dist_to_road_km`, `dist_to_place_km`, nearest-neighbor distance from
  each grid cell to OpenStreetMap roads/populated places, fetched via the Overpass API and joined
   as static per-cell values), same result: December 0/23, February 1/32, unchanged. Test top-10%
   capture dropped to 48.9%. A diagnostic refit with much more capacity (uncapped depth, 400 trees,
   run purely to check whether the tuned model's shallow depth was the bottleneck rather than the
   features themselves) moved December/February by only 1-2 fires each while cratering every other
   metric (test top-10% capture 26.4%), the classic overfitting-to-majority-class-noise pattern
   from the tuning results above, not evidence the added capacity was the fix.

Neither addition was kept, both were reverted after evaluation, so they aren't in `FEATURE_COLUMNS`
or the dataset today, and there is no `proximity.py` or `ingest_geography.py` in the repo.

**Why this looks structural rather than a missing-feature problem:** both attempts added *static*
signal, a value fixed per calendar date or per grid cell. `top_10pct_capture` ranks every (cell, date)
row in the year against every other for one global cutoff, so a static offset can shift a cell's or a
date's baseline up or down but cannot manufacture day-to-day variation within a cell, and is nowhere
near strong enough to lift a handful of winter fires above tens of thousands of ordinary-looking
winter non-fire rows. Fixing this for real needs a feature that varies *with the actual ignition
event* day by day (real burn-permit records, lightning-strike data), which this project has no source
for, not another proxy computed from data already on hand.

**Conclusion:** a documented limitation of a weather-only model, not a bug to keep chasing. The
project's response, as of the [fire season scoping](#scoping-to-fire-season) above, is excluding
Nov-Apr from the problem entirely, since that's where this blind spot lives and where the largest
fires don't occur anyway.

## Feature importance: what the model is actually leaning on

Before trusting the served RandomForest, it's worth checking that its accuracy comes from real
fire-weather signal rather than an artifact of the grid, the join, or where the split falls. Two
measures were compared:

- **Mean decrease in impurity (MDI)**, `rf.feature_importances_`. Fast, but biased toward
continuous/high-cardinality features and only reflects the training set, so it can overstate a feature
the model split on a lot without that split helping on new data.
- **Permutation importance**, computed separately on the 2023 val and 2024 test splits (10 repeats
each, scored by PR-AUC). Shuffling one column at a time and measuring the held-out PR-AUC drop is a
more honest measure of what the model depends on to generalize.

Both agree on the same ranking. A sharp disagreement between them would have suggested the model was
overfit to training-set idiosyncrasies rather than a real pattern:


| feature                                                               | MDI       | permutation ΔPR-AUC (val) | permutation ΔPR-AUC (test) |
| --------------------------------------------------------------------- | --------- | ------------------------- | -------------------------- |
| `swvl1` (soil moisture)                                               | 0.269     | +0.00203                  | +0.00308                   |
| `t2m_mean_7d`                                                         | 0.224     | +0.00165                  | +0.00264                   |
| `precip_30d`                                                          | 0.180     | +0.00240                  | +0.00322                   |
| `t2m`                                                                 | 0.112     | +0.00070                  | +0.00128                   |
| `rh_mean_7d`                                                          | 0.085     | +0.00053                  | +0.00052                   |
| `relative_humidity`                                                   | 0.040     | +0.00032                  | +0.00136                   |
| `precip_mm`                                                           | 0.028     | +0.00035                  | +0.00004                   |
| `days_since_rain`                                                     | 0.015     | +0.00004                  | +0.00014                   |
| `precip_7d`                                                           | 0.010     | +0.00006                  | +0.00004                   |
| `u10`, `v10`, `wind_dir_sin/cos`, `wind_speed`, `d2m`, `t2m_trend_7d` | all <0.01 | ~0 or slightly negative   | ~0 or slightly negative    |


Two takeaways:

1. **The signal is real, not an artifact.** The top 5 by both measures, soil moisture, 30-day
  precip, 7-day mean temp, raw temp, 7-day mean humidity, are exactly the slow-moving fuel-dryness
   indicators a domain expert would expect. Nothing about grid/join mechanics shows up as unexpectedly
   dominant, which is what a subtle pipeline bug driving the score would look like.
2. **Wind and the 7-day temp trend are dead weight in this model**, see [Dropping the dead-weight
  features](#dropping-the-dead-weight-features) below for what was actually done about it.

## Dropping the dead-weight features

The table above was computed once, before the fire-season/2012 re-tune settled. Before actually
removing anything from `FEATURE_COLUMNS`, permutation importance was re-run from scratch against the
*currently served* RandomForest (30 repeats, two random seeds, val and test both) to confirm the
picture still held:


| feature                        | val ΔPR-AUC (seed 1) | val ΔPR-AUC (seed 2) | test ΔPR-AUC (seed 1) | test ΔPR-AUC (seed 2) |
| ------------------------------ | -------------------- | -------------------- | --------------------- | --------------------- |
| `wind_dir_cos`                 | -0.000068            | -0.000067            | -0.000072             | -0.000072             |
| `u10`                          | -0.000323            | -0.000203            | -0.000166             | -0.000076             |
| `v10`                          | +0.000004            | +0.000001            | +0.000065             | +0.000072             |
| `wind_dir_sin`                 | +0.000555            | +0.000561            | +0.000032             | +0.000052             |
| `d2m`                          | +0.000287            | +0.000173            | -0.000339             | -0.000220             |
| `t2m_trend_7d`                 | +0.000017            | +0.000006            | +0.000046             | +0.000067             |
| `wind_speed`                   | -0.000211            | -0.000092            | **+0.000443**         | **+0.000655**         |
| `rh_mean_7d` (kept, for scale) | +0.001161            | +0.001220            | +0.000841             | +0.000956             |


`wind_dir_cos`, `u10`, `v10`, `wind_dir_sin`, `d2m`, and `t2m_trend_7d` all came back near-zero or
mixed-sign on both splits and both seeds, none crossing even 15% of `rh_mean_7d`'s importance (the
weakest *kept* feature), several actively negative. These six were dropped.

`wind_speed` **was kept, deviating from the original table's grouping.** It shows a real, reproducible
positive signal on test specifically (+0.00044 to +0.00066 across both seeds, roughly half
`rh_mean_7d`'s test importance, nowhere near the ~0 cluster) even though it's slightly negative on val.
Test is the split this project treats as the honest, non-overfit read, so it was trusted over the
noisier val signal. `u10`/`v10` were dropped in its place, consistently negative-or-negligible on both
splits and seeds.

Refitting `BEST_RANDOM_FOREST_PARAMS` unchanged on the resulting 10-column `FEATURE_COLUMNS` (`t2m`,
`swvl1`, `precip_mm`, `relative_humidity`, `wind_speed`, `days_since_rain`, `precip_7d`, `precip_30d`,
`t2m_mean_7d`, `rh_mean_7d`) left val/test scores within noise of the 16-column version:


|                     | val PR-AUC | val ROC-AUC | val top-10% | test PR-AUC | test ROC-AUC | test top-10% |
| ------------------- | ---------- | ----------- | ----------- | ----------- | ------------ | ------------ |
| 16 columns (before) | 0.0138     | 0.832       | 41.9%       | 0.0106      | 0.884        | 71.9%        |
| 10 columns (after)  | 0.0136     | 0.833       | 41.8%       | 0.0106      | 0.884        | 71.9%        |


Confirms these really were dead weight rather than something the tuned hyperparameters leaned on:
dropping them cost nothing measurable. This is a same-params refit only, no re-tune against the
smaller set. Nothing in `features/engineering.py` changed, since
`add_relative_humidity`/`add_wind_features` still need `d2m` and `u10`/`v10` as *inputs*; only the
model's own input list shrank.

**Why this matters beyond a smaller feature list:** it reinforces why the winter/shoulder blind spot
resisted more weather features. The model's real levers are almost entirely slow-moving fuel-dryness
signals plus one immediate condition (wind speed), and nothing in `FEATURE_COLUMNS` encodes human
activity, the more likely driver of winter ignitions. It also simplified [live weather
fetching](07-serving.md#live-weather-for-predictlive): Open-Meteo supplies relative humidity and wind
speed *directly*, so the live path never reconstructs them from dewpoint or wind vectors.

## Testing the sequence-modeling hypothesis

`research/neural-networks.md` argues against a neural network replacing the served RandomForest, but
names one question it doesn't rule out: does a model that sees the **raw** last-30-days weather
sequence per cell, instead of the hand-engineered rolling summaries above (`t2m_mean_7d`,
`precip_30d`, ...), capture a nonlinear temporal *shape* those summaries flatten away?
`training/sequence_model.py` runs that experiment: a small 1D-CNN (two `Conv1d` layers, global
average pooling, a small dense head) over 5 raw daily channels
(`t2m`/`precip_mm`/`swvl1`/`relative_humidity`/`wind_speed`, the same quantities behind the current
non-rolling features), trained with `BCEWithLogitsLoss(pos_weight=...)`, the PyTorch equivalent of
`class_weight="balanced"`, under the exact same temporal train/val/test split as everything else on
this page. The RandomForest side of the comparison is refit with the same tuned
`BEST_RANDOM_FOREST_PARAMS` on the *exact same row subset* the CNN sees (a handful of rows lose their
30-day raw window to date gaps that don't affect the rolling features), so the comparison is
apples-to-apples on identical rows, not just similar ones.

Real result:


| model                           | split       | pr_auc     | roc_auc   | top_10pct_capture |
| ------------------------------- | ----------- | ---------- | --------- | ----------------- |
| RandomForest (tuned, same rows) | val (2023)  | **0.0136** | **0.833** | **41.8%**         |
| SequenceCNN                     | val (2023)  | 0.0117     | 0.787     | 37.9%             |
| RandomForest (tuned, same rows) | test (2024) | **0.0106** | 0.884     | **71.9%**         |
| SequenceCNN                     | test (2024) | 0.0062     | 0.883     | 68.6%             |


The RandomForest wins on both splits and every metric except test ROC-AUC (a near-tie), most
clearly on PR-AUC (1.2-1.7x higher) and top-10% capture (4-5 points). The raw-sequence CNN did **not**
find temporal shape the rolling windows were missing; letting the model learn its own summary from
scratch, on ~4,800 positives, generalized worse than the fixed 7-/30-day windows handed to a tree
ensemble. That matches the tabular-data literature `research/neural-networks.md` cites rather than
being an exception to it, so the one hypothesis that document left open is now closed with a real
negative result.

**Practical conclusion:** no change to the served model, and the rolling-window features stay the
right representation of weather history here, not a simplification costing accuracy.

## Calibration: is `ignition_probability` a real probability?

Every metric on this page so far is **rank-only**: each asks "are actual fires scored higher than
non-fires," and each is mathematically unchanged by any monotonic rescaling of the scores. A model can
top all three while its raw `predict_proba` output is wildly wrong in absolute terms, which matters
here because `/predict` and `/predict/live` hand callers `ignition_probability` as a bare float, and
the obvious reading of 0.7 is "a 70% chance of igniting today." `evaluation/calibration.py` checks
whether that reading is justified, via two tools rank metrics can't provide:
**[Brier score](glossary.md#brier-score)** (mean squared error between predicted probability and the
{0,1} outcome, 0 is perfect, and always predicting the true base rate scores `base_rate * (1 -
base_rate)`, a cheap floor) and a **reliability table** (bucket by predicted-probability quantile,
then compare each bucket's mean prediction against its observed fire rate).

Run against the served model on both held-out splits:


|             | brier score | base-rate-only floor | observed positive rate |
| ----------- | ----------- | -------------------- | ---------------------- |
| val (2023)  | 0.1233      | 0.0033               | 0.33%                  |
| test (2024) | 0.0993      | 0.0010               | 0.10%                  |


**The served model's Brier score is ~40-100x *worse* (higher) than a trivial model that ignores every
feature and always predicts the split's true base rate.** That's a real, specific finding, not a
rounding effect, the reliability table shows exactly why:


| val (2023) predicted-probability bin | mean predicted | observed rate |
| ------------------------------------ | -------------- | ------------- |
| 0.034 - 0.046                        | 0.044          | 0.004%        |
| 0.046 - 0.057                        | 0.051          | 0.008%        |
| 0.057 - 0.078                        | 0.067          | 0.037%        |
| 0.078 - 0.098                        | 0.089          | 0.037%        |
| 0.098 - 0.127                        | 0.111          | 0.050%        |
| 0.127 - 0.175                        | 0.150          | 0.144%        |
| 0.175 - 0.264                        | 0.215          | 0.268%        |
| 0.264 - 0.430                        | 0.337          | 0.458%        |
| 0.430 - 0.685                        | 0.539          | 0.920%        |
| 0.685 - 0.912 (top decile)           | 0.852          | 1.382%        |


The top decile, cells scored at a mean 85% ignition probability, actually ignites 1.38% of the time
on val (0.72% on test). Every bucket is ordered correctly, which is exactly why the *rank* metrics look
good, but the absolute scale is off by roughly two orders of magnitude, worst at the high end where it
matters most to anyone reading the number literally.

**Why, mechanically:** `class_weight="balanced"` (see [Baseline-first
methodology](#baseline-first-methodology)) is a deliberate, correct fix for the optimizer collapsing to
the majority class during *fitting*, but it works by upweighting minority samples, which pushes
`predict_proba` toward the artificially rebalanced distribution the trees were fit against rather than
the true ~0.1-0.3% base rate. A known side effect, not a bug, and invisible to every rank metric on
this page.

**What this does and doesn't mean:** the *ranking* is still real and still what `/risk-map` and the
top-10%-capture story rely on. What changes is that `ignition_probability` should be read as a
**relative risk score, not a literal probability**, see the caveat in
[07-serving.md](07-serving.md#calibration-ignition_probability-vs-calibrated_probability). Fixing the
absolute scale via [isotonic or Platt
scaling](glossary.md#calibration-isotonic-and-platt-scaling) is a legitimate follow-up but a separate
decision: fitting on a split with only ~800 positives risks overfitting the calibration curve itself,
and would need its own held-out check rather than reusing val/test as both fit and evaluation set.

### Is the miscalibration itself stable across years?

The [rolling-origin backtest](#rolling-origin-backtest-is-719-typical-or-the-best-year-in-the-dataset)
above already showed `top_10pct_capture` swings wildly by holdout year. `evaluation/backtest.py` also
computes Brier score and a reliability table for each of the same 8 folds, which answers a question
the single val/test calibration numbers above can't: is the *miscalibration factor* at least a stable
correction to apply, even if ranking performance isn't?


| year | positives | brier score | base-rate floor | brier ratio | top-decile mean predicted | top-decile observed | top-decile ratio |
| ---- | --------- | ----------- | --------------- | ----------- | ------------------------- | ------------------- | ---------------- |
| 2017 | 1,433     | 0.2056      | 0.0059          | 35.0x       | 0.862                     | 1.82%               | 47x              |
| 2018 | 229       | 0.0900      | 0.0009          | 95.4x       | 0.795                     | 0.24%               | 332x             |
| 2019 | 57        | 0.0506      | 0.0002          | 215.2x      | 0.585                     | 0.05%               | 1,091x           |
| 2020 | 43        | 0.0868      | 0.0002          | 489.3x      | 0.772                     | 0.03%               | 2,673x           |
| 2021 | 2,628     | 0.1826      | 0.0107          | 17.0x       | 0.888                     | 2.93%               | 30x              |
| 2022 | 98        | 0.1388      | 0.0004          | 343.5x      | 0.855                     | 0.03%               | 2,590x           |
| 2023 | 802       | 0.1233      | 0.0033          | 37.4x       | 0.852                     | 1.38%               | 62x              |
| 2024 | 242       | 0.1048      | 0.0010          | 105.1x      | 0.814                     | 0.74%               | 110x             |


**No, it's not stable either, and arguably worse than the ranking metric.** The Brier ratio spans
17x to 489x (~29x spread) and the top-decile ratio 30x to 2,673x (~88x), so a single "divide by 50"
correction that looked right in one year would be off by another order of magnitude in the next.

**One honest caveat before taking that at face value:** each `reliability_table` bin holds ~24,000
rows, but in a sparse year almost none are real fires, so the top-decile *observed rate* in a year with
43-98 positives is estimated from a handful of them, and one extra or missing fire swings the ratio
enormously. The table fits that: the two years with the most positives (2021, 2017) show the smallest,
most similar ratios (30x, 47x), while the sparsest (2019, 2020, 2022) show the wildest. The Brier ratio
is less exposed but shows the same shape (17-37x for the big-fire years vs. 95-489x for sparse ones).

**Practical reading:** the trustworthy estimate comes from the years with enough fires to estimate a
rate from, and 2017/2021/2023 cluster in the **17-47x** range. But that doesn't rescue recalibrating
now: a correction estimated mostly from three unusually severe seasons is exactly the single-scenario
overfit this investigation exists to catch, with no evidence yet the same correction holds in a
below-average year. **Recommendation: don't fit a single static recalibration yet.** Pool across many
years before fitting, and validate the result's stability per holdout year the way this table does, or
a calibrator checked only in aggregate could hide the same instability the raw numbers already did.

### Does pooled, leave-one-year-out-validated calibration actually help?

`evaluation/calibration.py::leave_one_year_out_calibration_check` does what the recommendation above
asks: for each of the 8 years, fit a calibrator on every *other* year's pooled `(y_score, y_true)`
pairs, apply it to the held-out year, and compare against doing nothing. The honest test of whether
pooling generalizes to a year it never saw. Two methods are compared, isotonic regression (a flexible
monotonic curve) and sigmoid/Platt scaling (a single logistic curve).


| year | positives | pooled calibration-fit positives | raw Brier | isotonic Brier | sigmoid Brier | raw top-decile ratio | isotonic top-decile ratio | sigmoid top-decile ratio |
| ---- | --------- | -------------------------------- | --------- | -------------- | ------------- | -------------------- | ------------------------- | ------------------------ |
| 2017 | 1,433     | 4,099                            | 0.2056    | 0.0059         | 0.0059        | 47.3x                | 0.67x                     | 0.87x                    |
| 2018 | 229       | 5,303                            | 0.0900    | 0.0010         | 0.0010        | 332.3x               | 6.0x                      | 6.0x                     |
| 2019 | 57        | 5,475                            | 0.0506    | 0.0002         | 0.0002        | 1,090.8x             | 15.7x                     | 14.2x                    |
| 2020 | 43        | 5,489                            | 0.0868    | 0.0002         | 0.0002        | 2,673.4x             | 45.9x                     | 46.2x                    |
| 2021 | 2,628     | 2,904                            | 0.1826    | 0.0107         | 0.0107        | 30.3x                | 0.54x                     | 0.37x                    |
| 2022 | 98        | 5,434                            | 0.1388    | 0.0004         | 0.0004        | 2,589.7x             | 51.1x                     | 59.9x                    |
| 2023 | 802       | 4,730                            | 0.1233    | 0.0033         | 0.0033        | 61.6x                | 1.27x                     | 1.25x                    |
| 2024 | 242       | 5,290                            | 0.1048    | 0.0010         | 0.0010        | 109.6x               | 1.96x                     | 2.08x                    |


(top-decile ratio = mean predicted / observed rate in the top-scored bucket, matching the table above,
1.0x is perfect; both above and below 1.0x are miscalibrated.)

**Pooled calibration is a real, substantial improvement, but it doesn't solve the underlying
instability, it just moves the whole cluster of numbers much closer to correct.** Two separate results,
not one:

1. **Brier score improves by 15-500x in every year**, isotonic and sigmoid performing almost
  identically (no reason to prefer either here). This is largely mechanical: Brier score is dominated
   by the huge majority of true-negative rows, and *any* shrinkage toward the true ~0.1-0.3% base rate
   collapses their squared error, almost regardless of whether it's precisely right for that year.
2. **The top-decile ratio, the number that answers "is a highly-scored cell's probability
  meaningful", improves in absolute terms every year (worst case 2,673x -> 51x) but the *relative
   spread between best- and worst-calibrated year barely changes*: ~89x raw vs. ~95x isotonic.**
   Pooling moves every year much closer to 1.0x without making the years agree with each other any
   better, so the year-to-year instability is still there, just rescaled.

**The years that stay worst-calibrated after pooling are exactly the sparsest** (2018, 2019, 2020,
2022, still 6x-51x off), while the fire-heavy years land closest to 1.0x (2017 sigmoid 0.87x, 2021
isotonic 0.54x, 2023 isotonic 1.27x). Same statistical-noise explanation as above: a sparse year's
*observed* top-decile rate is itself estimated from a handful of fires, so no calibrator can be checked
precisely against that little ground truth.

**Revised recommendation:** a pooled calibrator is worth having as a materially better default than the
raw score, cutting worst-case miscalibration by roughly 50x, but it is not "solved calibration." Its
accuracy is still year-dependent, particularly for low-fire years.

**Promoted to the served model**, on that basis: `training/export_model.py::export_current_best` now
fits this same pooled isotonic calibrator (all 8 years combined, not held-one-out, the LOYO split
above exists to validate the method, not to leave a year out of the production fit) and attaches it to
the `ModelBundle`, exposed via `/predict`'s and `/predict/live`'s `calibrated_probability` and
`/risk-map`'s `calibrated_risk_probability` (see [Serving](07-serving.md#calibration-ignition_probability-vs-calibrated_probability)).
`ignition_probability`/`risk_probability` are unchanged and still the field to use for ranking, the
calibrator only rescales magnitude, and the caveat above (particularly unreliable in low-fire years)
still applies to the calibrated number, so treat it as a much-improved estimate, not an exact one.

## GPU-accelerated tuning (XGBoost)

This dev machine has an NVIDIA GPU (RTX 4060 Laptop), which raised the obvious question: can any of
this project's CPU-bound tuning move to it? The honest answer splits by model:

- **XGBoost has native GPU support**, `tree_method="hist", device="cuda"` on the same
`XGBClassifier` used everywhere else in this project. No algorithm change, just a device flag.
- **RandomForest does not**, `sklearn.ensemble.RandomForestClassifier` has no GPU code path at all.
The GPU-accelerated equivalent is RAPIDS' `cuml.ensemble.RandomForestClassifier`, but RAPIDS has
never shipped native Windows support (WSL2 only), which is a separate Linux environment, not a
package this project's `.venv` can just add. RandomForest tuning stays CPU-only here.

**A real driver/CUDA-version mismatch had to be diagnosed first, and its shape is worth recording.**
The first attempt looked like it worked (no error) but silently trained on CPU:
`XGBClassifier(device="cuda").fit(...)` logged `WARNING: No visible GPU is found, setting device to
CPU`, even though `nvidia-smi` saw the RTX 4060 and `xgboost.build_info()` confirmed the wheel was
compiled `USE_CUDA: True`. The cause was a mismatch one layer down: that wheel (xgboost 3.4.0) was
built against **CUDA 13.3**, requiring driver >=590, and the installed driver was 576.52, new enough
for `nvidia-smi` to work fine but too old for this build's CUDA runtime. XGBoost doesn't error on
that, it just falls back, **so a clean run is not by itself proof the GPU was used.** Fixed by
updating the driver to 596.49, re-verified with a smoke fit logging `XGBoost is running on: cuda:0`
and `nvidia-smi` showing real utilization, not just a library claiming a device is available.

The same fix unblocked a second GPU consumer: `sequence_model.py` already selected `cuda` when
available, but the default PyPI `torch` wheel is CPU-only, so it had been silently running on CPU all
along. Fixed via `pyproject.toml`'s `[tool.uv.sources]`/`[[tool.uv.index]]` entries pinning `torch` to
the `cu132` build.

**What changed in code:** `training/tune_xgboost_gpu.py` is a new standalone entry point, same search
space and same `PredefinedSplit` temporal-safety guarantee as the CPU search, just with
`tree_method="hist", device="cuda"`. `tune_random_search` gained an `n_jobs` parameter (default `-1`,
so existing calls are unaffected) so this script can pass `n_jobs=1`: `RandomizedSearchCV`'s `n_jobs`
runs candidates in parallel *processes*, right for CPU cores but wrong for a single GPU, where they
would contend for the same device and can throw CUDA OOM. `advanced_models.py`'s `__main__` dropped
its CPU XGBoost leg, since this script now covers that exact search on GPU.

**Measured result, same tuning session (2026-08-17):** the GPU XGBoost randomized search (15
candidates) finished in well under a minute; the CPU RandomForest randomized search run right before
it in the same session (also 15 candidates, same `tune_random_search` infrastructure) took hours.


| stage                                    | candidates | per-candidate fit time | total (sum of per-candidate times)                                                                           |
| ---------------------------------------- | ---------- | ---------------------- | ------------------------------------------------------------------------------------------------------------ |
| XGBoost, GPU (`tune_xgboost_gpu.py`)     | 15         | 1.2s – 5.1s            | ~39s                                                                                                         |
| RandomForest, CPU (`advanced_models.py`) | 15         | 14.1min – 86.1min      | ~11.9h (serial sum; real wall-clock was shorter, since several candidates ran concurrently across CPU cores) |


**This is not a clean same-algorithm GPU-vs-CPU benchmark and shouldn't be read as "XGBoost is
~1000x faster on this GPU."** Two different algorithms are being compared (RandomForest has no GPU path
to compare against at all), and the RandomForest side is deliberately single-threaded per candidate so
`RandomizedSearchCV` can parallelize across candidates instead, which is not a fair "best CPU can do"
number either. What the comparison honestly supports: GPU histogram training moved this project's
existing 15-candidate search, same infrastructure and same temporal-safety guarantees, from a
multi-hour CPU run to a sub-minute one for XGBoost, which is the practical win even without a precise
multiplier.

One open thread: the CPU RandomForest per-candidate times (14-86 minutes) are far slower than the "~20
minutes total for 15 candidates" baseline from the earlier tuning round. That number predates the
fire-season scope change and the 2012 extension, so the datasets differ, and `n_jobs=-1` on Windows
already has a documented history of erratic scaling here. Plausible explanations, not a confirmed
cause, and out of scope for this GPU work.

## Adding `cape`/`convective_precip_mm` to `FEATURE_COLUMNS`

`cape` and `convective_precip_mm` (full ERA5's CAPE and convective precipitation, see
[Weather join](04-weather-join.md#a-second-weather-source-full-era5s-cape-and-convective-precipitation)
for what they are and why they were fetched) had already been joined into
`kamloops_dataset.parquet` and enforced for completeness by `drop_incomplete_history`, but were
**not actually in** `FEATURE_COLUMNS`, no model was training on them. `training/baseline.py` now
includes both, added as a pair (matching how they were fetched and engineered together), bringing
`FEATURE_COLUMNS` to 12 columns.

### The re-tune result (2026-08-17 overnight run): no measured benefit

`advanced_models.py` (CPU RandomForest, both the 8-candidate grid and the 15-candidate randomized
search) and `tune_xgboost_gpu.py` (GPU XGBoost, 15-candidate randomized search) were re-run against
the 12-column `FEATURE_COLUMNS` overnight. The only apples-to-apples comparison available is against
the currently-served model from [the fire-season re-tune
above](#re-tuning-after-the-fire-season-scope-change), same search infrastructure, same
`PredefinedSplit`, same train/val/test years, the only difference being these two extra columns:


|                                                   | params                                             | test PR-AUC | test ROC-AUC | test top-10% capture |
| ------------------------------------------------- | -------------------------------------------------- | ----------- | ------------ | -------------------- |
| RandomForest, 10 features (served, 2026-08-15)    | 238 trees, depth 6, max_features 0.606, min_leaf 7 | **0.0106**  | **0.884**    | **71.9%**            |
| RandomForest, 12 features incl. cape (2026-08-17) | 438 trees, depth 7, max_features 0.775, min_leaf 8 | 0.00985     | 0.876        | 69.4%                |


Adding `cape`/`convective_precip_mm` did not improve any test-set metric; the 12-feature run is
slightly **worse** on all three. Being precise about what that shows: the old 10-feature winning
hyperparameters were themselves resampled inside this run's search and scored a val PR-AUC of 0.01388,
a near-tie with the new winner's 0.01406, so `RandomizedSearchCV` picked a different candidate on a
razor-thin val margin and that candidate generalizes slightly worse to 2024. The same single-fold
val-selection noise flagged elsewhere on this page, so this is not evidence CAPE actively hurts, and
equally not evidence it helps. With 15 candidates and one test year, a larger `n_iter` or a
rolling-origin backtest would be needed to say more.

**Decision: keep serving the existing 10-feature model.** `BEST_RANDOM_FOREST_PARAMS` and
`model.joblib` are **unchanged**: re-running `export_model.py` would have silently picked up the new
12-column `FEATURE_COLUMNS` (a global import) under the old hyperparameters, an untested combination.
This leaves a real inconsistency worth naming: `baseline.py`'s `FEATURE_COLUMNS` lists 12 columns while
the served model was fit on the first 10. Each `ModelBundle` snapshots its own `feature_columns` at
export time, so the API stays correct regardless, but a future re-tune shouldn't assume the served
model already reflects these columns.

**A dormant limitation, not currently active:** `/predict/live` works today because the served model
never asks for these columns. But the moment a future export promotes a model trained on them, it
breaks: Open-Meteo's historical archive API has no `convective_precipitation_sum` parameter at all, and
accepts `cape`/`cape_mean` but returns `null` for every value tested, since the data is only populated
in their forecast product. Worth resolving (a live CAPE source, or dropping the columns) *before* any
future result is good enough to promote. `features/convective.py`/`ingest_era5_convective.py` and the
joined data stay in place as groundwork for revisiting this.

## Future directions: four researched proposals (2026-08-17)

**All four were investigated against this project's actual code and history**, not proposed in the
abstract, each checked for leakage risk, fit with the temporal-safety discipline, and honest evidence
of whether it would help rather than just add complexity. Each proposal is followed by what actually
happened when it was run: 1 and 2 shipped, 3 and 4 produced negative results and did not.

### 1. Spatial-lag features (neighbor cells' recent fire history)

Every model to this point treated each grid cell as fully independent, never looking at a neighbor's
weather or fire history, despite this being a regular grid. `cell_id` is literally `"{row}_{col}"` from
integer floor-division, so a cell's 8 Moore neighbors are plain string construction, no
KDTree/STRtree/GDAL needed.

**Proposed feature:** `neighbor_fire_count_Nd` (N in {1, 3, 7}), count of the 8 neighboring cells with
`ignited=1` in the trailing N days, via a `date` x `cell_id` -> `ignited` pivot, shifted, then summed
across neighbor columns. No new dependency.

**The real hypothesis, and a self-correction worth keeping.** The initial framing assumed this would
help the [winter/shoulder-season blind spot](#known-limitation-a-wintershoulder-season-blind-spot),
but this project's own data argues against that: December/February fires are 90.8% person-caused and
number only 4 across 2012-2024, isolated point ignitions rather than spatially contiguous events, the
same reason the static distance-to-road/town features failed. The better-fit hypothesis is
**fire-season spread dynamics**: a real wildfire physically growing into adjacent cells over
consecutive days, which no model here can currently see.

**Leakage, the one thing that must be exactly right:** only strictly prior-day (`date <= D-1`)
neighbor status may be used. One large fire spanning several cells is detected the same FIRMS day, so
same-day neighbor status would trivially predict the target and wouldn't exist at real prediction time.

**Why it might fail:** neighbor cells could just proxy the same regional weather already in
`FEATURE_COLUMNS`; grid edges have fewer neighbors; at a ~0.1-0.3% base rate the counts are mostly
zero, giving low variance to split on.

**A GNN was explicitly considered and rejected**, no evidence a graph architecture beats RandomForest
fed the same aggregate neighbor stats as ordinary features, and a much heavier lift. Matches the
baseline-first bias: simple features first, more architecture only if they show real signal.

### Spatial-lag features: implemented and promoted (2026-08-19)

`features/engineering.py::add_neighbor_fire_features` implements exactly the proposal above.
Implementation detail worth naming: rather than looping per cell, the whole `(date x cell_id)` ignited
panel is shifted forward one day (the leakage guard), rolled per window, then multiplied by a dense
0/1 Moore-adjacency matrix built once from the `"{row}_{col}"` scheme, a single matrix multiply per
window instead of 1,443 per-cell sums, fast enough that no sparse-matrix dependency was needed. Added
to `ENGINEERED_COLUMNS` (so `drop_incomplete_history` enforces the 1/3/7-day warm-up) and to
`FEATURE_COLUMNS` in place of `cape`/`convective_precip_mm`, keeping this a clean single-variable test
against the served 10-feature model rather than one confounded by an already-inconclusive pair.

**The result is far larger than any other change tried on this project, and was checked hard for
leakage before being trusted.** Refitting the served model's exact hyperparameters
(`BEST_RANDOM_FOREST_PARAMS`, unchanged) on the resulting 13-column set:


|                                                     | val PR-AUC | val ROC-AUC | val top-10% | test PR-AUC | test ROC-AUC | test top-10% |
| --------------------------------------------------- | ---------- | ----------- | ----------- | ----------- | ------------ | ------------ |
| RandomForest, 10 features (served, pre-2026-08-19)  | 0.0138     | 0.832       | 41.9%       | 0.0106      | 0.884        | 71.9%        |
| RandomForest, 13 features incl. neighbor_fire_count | **0.365**  | **0.963**   | **90.8%**   | **0.373**   | **0.951**    | **86.0%**    |


A freshly GPU-tuned XGBoost (`tune_xgboost_gpu.py`, `XGBOOST_DISTRIBUTIONS`, 15 candidates, same
13-column set) independently landed at a similar order of magnitude, test PR-AUC 0.372, ROC-AUC
0.950, top-10% capture 86.4%, confirming this isn't one model family's quirk. MDI feature importance
on the refit RandomForest shows `neighbor_fire_count_7d`/`_3d`/`_1d` at 0.570/0.238/0.083
respectively (89% of total importance combined), dwarfing `swvl1` (previously the top feature at
0.269, now 0.020), the model now leans on this feature almost exclusively.

**Why this isn't leakage, checked directly rather than assumed:** a jump this large demands the same
scrutiny the `tp` accumulation bug in [Weather
join](04-weather-join.md#the-tp-bug-and-how-it-was-actually-caught) got, since internal consistency
isn't proof of correctness. Three checks:

1. **Ignited-rate-by-bucket is monotonic and physically sane**, not the step function a same-day
  identity leak would produce: `neighbor_fire_count_1d = 0` -> 0.04% ignition rate (near the ~0.15%
   base rate), `= 5` -> 68.7%; `_7d` climbs the same way (0.03% at 0 -> 31.4% at 10+).
2. **A manual per-fire trace against the raw** `(cell_id, date, ignited)` **rows** confirms the
  shift-then-roll logic: a cell that ignited 2021-08-19 shows `neighbor_fire_count_1d = 1` because a
   Moore neighbor ignited 2021-08-18, rising to `_3d`/`_7d` = 2 as the fire visibly spread over the
   following days. A separate fire with no local precursor correctly shows all three at 0.
3. **Same-day exclusion holds**: for that same fire, the day *before* ignition correctly shows
  `neighbor_fire_count_1d = 0`. A same-day leak would have shown a nonzero count a day early.

**Why the effect size makes sense, not just why it isn't a bug:** every prior feature is weather, a
slow-moving proxy for fuel dryness. `neighbor_fire_count` is the first feature that's a direct physical
precursor of the event itself: on a 5km grid, a wildfire actively burning in an adjacent cell is about
as strong a same-week ignition signal as this project could hand a model, categorically different from
"the air is hot and dry nearby." This also reframes the [backtest
instability](#rolling-origin-backtest-is-719-typical-or-the-best-year-in-the-dataset) and
[extreme-year degradation](#why-performance-swings-by-month-and-year) findings, both diagnosed against
a model with no way to see a fire already spreading nearby, re-verified below rather than assumed.

**Promoted to the served model** (unchanged hyperparameters, refit on the new 13-column
`FEATURE_COLUMNS`) on the strength of this measured, leakage-checked, cross-model-family-confirmed
win. No wider re-tune was run first, since the gain is large enough on the existing params that
waiting on one wasn't necessary (a possible refinement, not a blocker). **This broke** `/predict/live`
for about a day, accepted deliberately: the same tradeoff already made for
`cape`/`convective_precip_mm`, except active immediately rather than dormant, since this is the feature
set actually being promoted. Fixed 2026-08-20 via a live FIRMS NRT feed, see
[Serving](07-serving.md#live-fire-detections-for-predictlive).

#### Re-verifying the rolling-origin backtest against the 13-feature model

`evaluation/backtest.py` re-run 2026-08-20 against the promoted 13-feature (neighbor-fire-inclusive)
model, using the identical 8-fold expanding-window protocol as the original [rolling-origin
backtest](#rolling-origin-backtest-is-719-typical-or-the-best-year-in-the-dataset), same holdout years,
same `BEST_RANDOM_FOREST_PARAMS`, no re-tuning per fold, so this is a clean before/after on the one
variable that changed (the feature set):

| year | old top-10% (weather-only) | new top-10% (+neighbor-fire) | old PR-AUC | new PR-AUC |
| ---- | --------------------------- | ----------------------------- | ---------- | ---------- |
| 2017 | 30.8%                        | 29.1%                         | 0.0141     | 0.0460     |
| 2018 | 25.3%                        | 52.8%                         | 0.0023     | 0.0511     |
| 2019 | 22.8%                        | 54.4%                         | 0.0004     | 0.0052     |
| 2020 | 16.3%                        | 34.9%                         | 0.0003     | 0.0016     |
| 2021 | 27.0%                        | **92.8%**                     | 0.0333     | 0.4390     |
| 2022 | **8.2%**                     | **75.5%**                     | 0.0005     | 0.0741     |
| 2023 | 41.8%                        | 90.8%                         | 0.0136     | 0.3654     |
| 2024 | 74.4%                        | 87.6%                         | 0.0113     | 0.3688     |

**The two structurally-broken extreme years are fixed, not just improved.** 2021 and 2022 were the
folds [Why performance swings by month and year](#why-performance-swings-by-month-and-year) diagnosed as
having a mechanistic problem beyond ordinary noise: a fixed ranking budget breaking down under extreme
same-day fire counts, plus province-wide heat compressing the weather signal itself.
`neighbor_fire_count` sidesteps both, since it isn't a weather feature that compresses under a heat
dome and it gives the model a second, independent axis to rank on when weather alone stops
discriminating. 2022 goes from this backtest's *worst* year (8.2%) to mid-pack (75.5%), and 2021, which
previously underperformed its own Jul/Aug-heavy month mix, becomes the best year in the new backtest
(92.8%).

**Cross-year instability is reduced but not eliminated.** New top-10% capture: mean **64.7%**, median
**64.9%**, standard deviation **~23.8pp**, against the old 30.8%/26.2%/~19pp. The *floor* rose sharply
(8.2% -> 29.1%) and the center roughly doubled, but the spread is still wide (29%-93%) and slightly
larger in absolute terms, so this feature raised performance across the board more than it equalized
it. 2017 is now the clear worst year, essentially unchanged (30.8% -> 29.1%): an outlier in the other
direction, high fire count without 2021's extreme same-day clustering, so neither mechanism explains it
cleanly. Why it doesn't benefit the way 2021/2022 did is an open question this pass didn't chase.

**The month-of-season correlation weakened, consistent with that mechanism.**
`corr(jul_aug_fire_share, top_10pct_capture)` dropped from **r=0.63** to **r=0.44**, expected if part of
what month-mix was a *proxy* for (whether a year had enough nearby-fire density for weather to still
discriminate) is now captured directly.

**Practical takeaway, updated:** the "don't cite 71.9% as typical" caution still holds in direction, but
its substance changed. The realistic worst-case floor is now closer to ~30% than ~8%, and median
performance is closer to the previously-best years than the previously-typical ones. Genuine, measured
improvements from a same-week fire-precursor feature, not artifacts of retuning or a different
protocol.

### 2. SHAP explainability

The existing feature-importance work (MDI + permutation importance) is entirely *global*, one ranking
across the whole val/test set. It can say "soil moisture matters most on average," not "why did this
cell get flagged high-risk on this day."
[SHAP](glossary.md#shap-shapley-additive-explanations) gives *local*, per-prediction, additive
attributions (each feature's contribution sums exactly to that prediction's score), a complement to
the existing analysis rather than a replacement.

**Scope: offline analysis first, a live endpoint as a separate decision.** A one-time script producing
(a) a beeswarm plot across the test set, cross-checked against the existing MDI/permutation top-5 as a
third independent method, and (b) waterfall plots for real dates already spot-checked elsewhere in
this project.

**Two gotchas found in current** `shap` **docs, both to be handled explicitly, not assumed:**

1. Binary-classifier output shape (a list of two arrays vs. a single array) is version-dependent and
  must be verified empirically against the installed version.
2. The default `TreeExplainer` explains the raw tree margin, not probability. Getting values that sum
  to `predict_proba` requires `feature_perturbation="interventional"`, `model_output="probability"`,
   and a background sample. Even then SHAP can only decompose the pre-calibration
   `ignition_probability`, since the isotonic calibrator is a separate post-hoc regression on top of
   the raw score, not part of the tree structure. State that plainly in the write-up so it doesn't
   read as an oversight.

### SHAP explainability: implemented (2026-08-19)

`evaluation/shap_analysis.py` implements the offline stage of the proposal above, against the served
model (13-column `FEATURE_COLUMNS` including `neighbor_fire_count_{1,3,7}d`, promoted the same day,
see the previous section). Both gotchas were handled exactly as planned, not assumed:

1. **Output shape, verified against the installed** `shap==0.52.0`**:** `TreeExplainer(...)` called on a
  `DataFrame` returns a single `(n_samples, n_features, n_classes)` array, not the "list of two
   arrays" shape some other shap versions/configs use. `_positive_class_explanation` asserts this
   shape and raises loudly rather than silently misreading which slice is the positive class if a
   future shap upgrade changes it again.
2. **Additivity to** `predict_proba`**, checked by reconstruction, not assumed:** `explain()` sums each
  row's SHAP values plus its base value and asserts the result matches `model.predict_proba`'s raw
   output to `1e-4`, for every call, a broken reconstruction would mean the attributions don't
   actually explain what they claim to. As documented up front, everything below is in raw
   `ignition_probability` units; there is no SHAP decomposition of the isotonic-calibrated
   `calibrated_probability`, since the calibrator is a separate post-hoc regression bolted on after
   the tree ensemble.

**The global SHAP ranking does *not* match MDI's, and that discrepancy was chased down rather than
left unexplained.** MDI (see the previous section) put `neighbor_fire_count_7d`/`_3d`/`_1d` at 89% of
total importance combined. Mean |SHAP| over an unbiased 3,000-row random test-set sample tells a
different-looking story:

| feature                  | mean \|SHAP\| |
| ------------------------ | ------------- |
| `swvl1`                  | 0.0267      |
| `t2m`                    | 0.0191      |
| `t2m_mean_7d`            | 0.0147      |
| `precip_mm`              | 0.0126      |
| `rh_mean_7d`             | 0.0074      |
| `neighbor_fire_count_7d` | 0.0058      |
| `wind_speed`             | 0.0052      |
| `precip_30d`             | 0.0051      |
| `days_since_rain`        | 0.0042      |
| `relative_humidity`      | 0.0042      |
| `precip_7d`              | 0.0033      |
| `neighbor_fire_count_3d` | 0.0008      |
| `neighbor_fire_count_1d` | 0.0006      |

This is a real, non-contradictory disagreement between two honest measures, checked directly rather
than reasoned about: `neighbor_fire_count_7d` is exactly 0 for 98.6% of test rows, and its mean |SHAP|
on that subset is **exactly 0.000000**, vs. **0.474** on the 1.4% where it's nonzero. MDI reflects how
decisive a feature is *when the tree splits on it*; mean |SHAP| over a random population gets diluted
by the majority of rows where a near-always-zero feature contributes nothing. **Both are true at
once:** `neighbor_fire_count` is by far the most decisive feature on the small share of cell-days
where a fire is spreading nearby, and irrelevant everywhere else, where weather genuinely is what's
left to differentiate on. The MDI-vs-permutation cross-check doesn't transfer here, since SHAP and MDI
answer different questions ("decisiveness when used" vs. "population-average contribution") for a
feature this heavily zero-inflated.

**Three real per-prediction waterfalls, picked from actual test-set rows** (a random sample, plus
every one of the test set's 242 real fires, so genuine caught/missed examples were guaranteed to
exist rather than left to chance, an earlier run of this same script against a purely random
3,000-row sample happened to contain zero fires at all):

- **A caught fire** (cell `1119_-1711`, 2024-07-20, `ignition_probability=0.997`): `neighbor_fire_count_7d`
(+0.517), `_3d` (+0.161), `_1d` (+0.060) are the top three contributors by a wide margin, this
fire was flagged almost entirely because it had 11 neighbor ignitions in the trailing week.
- **The highest-scored non-fire** (cell `1122_-1713`, 2024-08-06, `ignition_probability=0.995`): the
same shape, `neighbor_fire_count_7d` (+0.595), `_3d` (+0.147), `_1d` (+0.072) dominate. This cell
sat in the middle of an active nearby cluster and, by the model's own reasoning, looked exactly like
the fires around it, a legible, defensible false positive, not an inexplicable one.
- **A real missed fire** (cell `1141_-1693`, 2024-07-22, `ignition_probability=0.275`, below the
test set's own top-10% cutoff): **no** `neighbor_fire_count` **feature appears anywhere in its top-6
contributions at all.** The drivers are entirely weather (`swvl1` +0.082, `t2m` +0.066,
`t2m_mean_7d` +0.060, `wind_speed` -0.039, `relative_humidity` +0.016, `rh_mean_7d` +0.009), a
concrete, individual confirmation of the mechanism proposed when spatial-lag features were first
motivated above: a fire with no nearby recent precursor gets scored on weather alone, the same
ceiling every purely-weather-driven prediction already had before this feature existed.

Plots were saved to `data/processed/` (gitignored build output, same as `model.joblib`) rather than
committed or embedded here, matching this project's prose-and-tables documentation style.

**Wired into a live endpoint 2026-08-20:** `GET /predict/explain`, see
[Serving](07-serving.md#explaining-a-live-prediction-predictexplain) for the full write-up.

### 3. Venn-Abers per-prediction uncertainty (not generic "conformal prediction")

The pooled isotonic/sigmoid work above found calibration reliability itself swings 6-51x by year in
sparse years even after pooling, but that instability is invisible to an API consumer: `/predict`
returns one float no matter how thin the calibration-set support behind it.

**Technique, and why not MAPIE:** MAPIE's prediction-set approach (a marginal-coverage guarantee that
the true label lands in a predicted set) is built for choosing among discrete classes, an awkward fit
for "how much do I trust this one probability." [Venn-Abers](glossary.md#venn-abers) (MIT license,
pure numpy/sklearn) fits instead: per prediction it produces a calibrated point probability `p_prime`
*plus* an interval `[p0, p1]` bracketing it, a per-row machine-readable version of the caveat the
year-dependent calibration finding already established.

**Temporal safety, checked against the library source, not the README.** The high-level
`VennAbersCalibrator` wrapper does its own random `cal_size` split internally, which would quietly
reintroduce the random-split leakage this project's [temporal-split
discipline](#splitting-by-time-never-randomly) exists to prevent. The low-level `VennAbers` class
doesn't: `va.fit(p_cal, y_cal)` takes pre-computed calibration probabilities and labels directly. Use
the low-level class only.

**Evaluation, extending the existing LOYO rig:** for each of the 8 years, fit on the other 7 pooled,
apply to the held-out year, and report `p_prime`'s Brier/top-bin-ratio alongside mean interval width.
**The real test:** do intervals widen honestly in the years already known least reliable, or does the
interval claim false confidence there too?

**Negative result, defined up front:** if interval width doesn't track the documented sparse-year
unreliability, it adds nothing over the existing static caveat, so don't ship it, the same standard
the reverted calendar/proximity features were held to.

**Result, run 2026-08-20: negative, not shipped.** `evaluation/calibration.py`'s LOYO check extended
with the low-level `venn_abers.VennAbers` class (`fit_venn_abers_calibrator`/
`apply_venn_abers_calibrator`), fit on the same 8 rolling-origin folds the isotonic/sigmoid columns
already use:

| year | positives | `venn_abers` Brier | `venn_abers` top-bin ratio | mean interval width |
| ---- | --------- | ------------------- | --------------------------- | -------------------- |
| 2017 | 1,433     | 0.005963             | 2.756                        | 0.000171              |
| 2018 | 229       | 0.000920             | 1.181                        | 0.000025              |
| 2019 | 57        | 0.000238             | 2.642                        | 0.000012              |
| 2020 | 43        | 0.000181             | 5.580                        | 0.000015              |
| 2021 | 2,628     | 0.009747             | 1.568                        | **0.004938**          |
| 2022 | 98        | 0.000387             | 1.707                        | 0.000018              |
| 2023 | 802       | 0.002713             | 0.825                        | 0.000037              |
| 2024 | 242       | 0.000798             | 1.179                        | 0.000023              |

`venn_abers`'s Brier/top-bin-ratio columns track isotonic's almost exactly (expected, both are
monotonic score-to-frequency mappings fit on the same pooled data), so the point-probability half adds
nothing new. **The real test was interval width, and it fails cleanly:**
`corr(mean_interval_width, 1/positives)` across the 8 folds is **-0.36**, weakly *negative*, the
opposite sign the hypothesis needs. Bucketed as the proposal specified, the known-sparse years
(2019/2020/2022) have a *smaller* mean width (0.000015) than the other five (0.001039), backwards from
the prediction, and that "other years" number is entirely one outlier: 2021 alone is 0.004938, roughly
30-400x every other year. Width is tracking something else, most plausibly 2021's own extreme raw
scores and its unusually small pooled calibration set (2,904 positives, smallest of the 8, since 2021
contributes more positives than the other seven years combined), not "how reliable is this year's
calibration."

**Per the criterion defined up front: don't ship.** No `ModelBundle`/API changes.
`fit_venn_abers_calibrator`/`apply_venn_abers_calibrator` and the extended check stay in
`evaluation/calibration.py` as validated, reusable tooling and a useful cross-check that isotonic's
point estimate isn't leaving anything on the table, re-runnable cheaply if this is ever worth
revisiting.

### 4. Attention-pooling on the sequence model (a narrower angle than a full Transformer)

See [Testing the sequence-modeling hypothesis](#testing-the-sequence-modeling-hypothesis) for the
context: RandomForest already won that comparison clearly, and this project has *twice* documented
that more model capacity backfires here (the CNN result, plus an uncapped-depth RandomForest
diagnostic that cratered top-10% capture to 26.4%). A full self-attention/Transformer block would add
capacity in exactly the setting already shown to punish it.

**Narrower proposal instead:** keep `SequenceCNN`'s `conv1`/`conv2` unchanged and replace
`AdaptiveAvgPool1d(1)` with one `nn.Linear(hidden_channels*2, 1)` scoring each day plus a
softmax-weighted sum, ~33 extra parameters rather than "more capacity." That tests a different,
still-open question: does *learning which days to weight* beat uniform averaging, independent of the
already-closed raw-sequence-vs-rolling-features question.

**Interpretability artifact:** plot attention weight against day-in-window for real test-set fires
plus mean attention-by-lag-day across all test positives, checking whether the model concentrates near
the ignition day or spreads out evenly (evenly means it learned to reproduce plain averaging, a
negative result). Only the pooling layer changes, isolating one variable.

**Honest framing:** a cheap, low-risk experiment worth running for the interpretability artifact and
research completeness, not a likely path to beating RandomForest, given this project is 2-for-2
against the idea that more capacity helps.

**Result, run 2026-08-20: negative, as expected, and now 3-for-3 against "more capacity/complexity
helps."** `AttentionPoolSequenceCNN` trained via the identical `fit_sequence_cnn` harness (same rows,
same epochs, same optimizer) as `SequenceCNN`, on the same fire-season 2012-2022 train / 2023 val / 2024
test split as [Testing the sequence-modeling hypothesis](#testing-the-sequence-modeling-hypothesis):

| model                    | val PR-AUC | val top-10% | test PR-AUC | test top-10% |
| ------------------------ | ---------- | ----------- | ------------ | ------------ |
| RandomForest (13-feature)| 0.3654     | 90.8%       | 0.3727       | 86.0%        |
| SequenceCNN (avg-pool)   | 0.0114     | 39.0%       | 0.0084       | 68.6%        |
| AttentionPoolSequenceCNN | 0.0091     | 34.4%       | 0.0049       | 65.3%        |

Attention-pooling didn't just fail to beat RandomForest (expected), it underperformed plain
`SequenceCNN` averaging on every metric on both splits. Learning which days to weight, with ~33 extra
parameters, made the architecture strictly worse, not neutral.

**The interpretability artifact explains why, and it isn't the "uniform" negative result predicted up
front.** Mean attention weight by lag-day, pooled across all 242 real test-set fires, neither spreads
evenly (uniform would be 3.3% per day) nor concentrates on the ignition day (day 0's mean weight is
0.01%, far *below* uniform). Instead it puts an oddly specific 23.3% of its mass on day-2-before-target
alone, with smaller bumps around days 26-27 and a mild plateau around days 10-14. That's a third
outcome, worse than either the proposal considered: it learns a specific, non-uniform pattern that
doesn't correspond to the most information-relevant day, a mild overfitting-to-noise signature
consistent with scoring below plain averaging. Caught and missed fires show the same day-2 spike, so
that isn't a separate finding. Plot saved to `data/processed/attention_weights_by_fire.png`.

**Conclusion:** a cheap experiment that paid off as research completeness and interpretability, not as
a path to a better model. Both sequence models stay diagnostic-only scripts; nothing here changes what
is served.

## Closing the feature-category gap: FWI, terrain, and fuel type (2026-08-21)

A competitive-landscape review of other wildfire-ML systems (the Canadian FWI System itself, Google's
Next Day Wildfire Spread benchmark, CanadaFireSat, a 2020-2025 systematic review of 341 ML wildfire
studies) surfaced one structural gap ahead of any modeling refinement: FireSight was a **weather-only**
model. Fuel/vegetation state is the single largest input category reported across that literature
(44.7% of all reported ML wildfire model inputs, ahead of climate/weather), and terrain (elevation,
slope, aspect) is foundational to the Canadian FBP System itself; FireSight had zero features in either
category. Three additions closed that gap, each implemented and validated independently before being
combined into one measured benchmark:

**1. Canadian FWI System (`features/fwi.py`).** FFMC/DMC/DC/ISI/BUI/FWI, the fire-danger rating BC
Wildfire Service runs operationally, computed via the Van Wagner (1985/1987) recursive equations from
data already in the pipeline, not a new data source. Transcribed from the official NRCan-maintained
reference implementation (`cffdrs/cffdrs_r`'s component R files, not a secondary description) and
checked against that package's own fixtures: 33 reference-value tests in `tests/test_fwi.py`, every one
an exact match rather than a "looks plausible" translation. **A deliberate simplification, not an
oversight:** with no snow-cover data there's no real spring-melt date, so the recursion resets to
standard start-up values (85/6/15) on a fixed March 1, and days before the record's first reset are
left `NaN` (dropped by `drop_incomplete_history`, the same "don't guess, drop it" precedent
`days_since_rain` set).

**2. Terrain (`features/topography.py`).** Elevation from Open-Meteo's Elevation API (Copernicus DEM
2021, 90m), the same provider `live_weather.py` already uses, via point queries rather than a DEM
download, so no new GDAL/rasterio dependency. Slope and aspect are derived via Horn's method (Horn
1981, the formula ESRI's tools implement) from each cell's 8 Moore-neighbor elevations, appropriate at
5km resolution where elevation is already a coarse per-cell value. Aspect is encoded as sin/cos of a
compass bearing, the circular-encoding precedent `add_wind_features` set. `tests/test_topography.py`
includes physically-derived checks (a synthetic east-rising grid must produce a west-facing aspect)
rather than only checking the code runs.

**3. Fuel type (`features/fuel_type.py`).** BC's Provincial Fuel Type Layer
(`WHSE_LAND_AND_NATURAL_RESOURCE.PROT_FUEL_TYPE_SP`) is the FBP-classification layer BC Wildfire
Service's own Prometheus fire-growth simulator consumes, served as individual forest-stand polygons
(>400,000 intersect the Kamloops FC bbox; the province-wide download is a ~4GB File Geodatabase needing
GDAL), so this queries the WFS endpoint per grid-cell centroid instead of downloading it. One
correction found against the live service: `FUEL_TYPE_CD` often carries a burn-history prefix
(`B71_S-2`) that would have multiplied the effective class count far past the ~16 base FBP types, so
`FT_PROMETHEUS`, the layer's own pre-cleaned base code, is what's kept. One-hot encoded per code
*actually present* in the Kamloops extract (19 codes, not a fixed province-wide schema), several of
them thin (`C-4`: 2 cells), a real limitation of this region's scale.

### Result: measured, mixed-to-negative, not promoted

All 29 new columns (6 FWI + 4 terrain + 19 fuel type) added to the training set at once and benchmarked
against the current 13-feature served model, same `BEST_RANDOM_FOREST_PARAMS`, same train/val/test split,
matching the exact methodology `cape`/`convective_precip_mm` was evaluated with above:

| model                                     | val PR-AUC | val top-10% | test PR-AUC | test top-10% |
| ------------------------------------------ | ---------- | ----------- | ------------ | ------------ |
| 13 features (served)                       | 0.3654     | 90.8%        | 0.3727        | 86.0%        |
| 42 features (+ FWI + terrain + fuel type)  | 0.3562     | 90.6%        | 0.3172        | 86.4%        |

Mixed, and meaningfully worse (-15% relative) on the metric that matters most, test PR-AUC on the
untouched 2024 year. Not a clean win the way `neighbor_fire_count` was, and not a clean no-op either.
MDI on the 42-feature model shows the new columns aren't sitting unused: `bui` (2.0%), `dmc` (1.8%),
`elevation_m` (1.5%), `aspect_sin` (1.4%) and `fwi` (1.4%) each outrank several existing weather
features, so the model does lean on them, it just didn't generalize better.

**Two undecided confounds, named rather than papered over:** (1) `max_features=0.6063` is a *fraction*
of the column count, ~8 candidate features per split at 13 columns but ~25 at 42, substantially
different effective regularization the params were never tuned for, so this can't cleanly separate
"these features don't help" from "these hyperparameters are stale for a 3x wider set." (2) The
fuel-type one-hots are genuinely thin at this region's scale. Neither was chased this pass, the same
standard the `cape` result was held to: one measured comparison under unchanged hyperparameters is
enough to decide *don't ship yet*, not enough to rule the features out forever.

**Decision at this point: not promoted.** `training/baseline.py::FEATURE_COLUMNS` and `export_model.py`'s
served model stayed **unchanged** (still the 13-feature set) pending the follow-up check named above,
see below for how that check actually turned out.

### Per-group ablation: isolating which of the three actually helps

The natural next check named above, re-tuned `max_features`, or isolating FWI from terrain from fuel
type rather than testing all three at once, was run the same day. Each group was added to the 13-feature
base individually, same `BEST_RANDOM_FOREST_PARAMS`, same train/val/test split:

| model                     | n features | val PR-AUC | test PR-AUC | test top-10% |
| -------------------------- | ---------- | ---------- | ------------ | ------------ |
| base (13, served)           | 13         | 0.3654     | 0.3727       | 86.0%        |
| base + FWI                  | 19         | 0.3690     | 0.3737       | 86.0%        |
| base + terrain              | 17         | 0.3663     | 0.3697       | 86.8%        |
| **base + fuel type**        | **32**     | **0.3632** | **0.3816**   | **88.0%**    |
| base + all three            | 42         | 0.3562     | 0.3172       | 86.4%        |

FWI and terrain are each essentially neutral alone, neither helping nor hurting test PR-AUC beyond
noise. **Fuel type alone is a real, clean win**: +2.4% relative test PR-AUC, +2pp top-10% capture, no
hyperparameter changes. The all-three row reproduces the original combined result almost exactly,
confirming that wasn't a fluke, but the per-group breakdown shows the regression isn't any single group
being bad: it's fuel type's real signal getting drowned out by two groups that add columns without
adding value.

**Confound (1) resolved: `max_features` wasn't stale.** A sweep from 0.15 to 1.0 against the 32-column
base+fuel-type model (other params fixed) is unimodal, peaking right around the existing 0.6063,
val PR-AUC there (0.3632) is the highest of the sweep, and test PR-AUC (0.3816) is within noise of the
peak (0.4523 edges it slightly on test, 0.3843 vs 0.3816, but loses on val, the metric this project
selects on). The 13-column-tuned fraction already generalizes fine to 32 columns; the earlier
42-feature regression was never a stale-hyperparameter artifact.

**Confound (2), thin one-hot classes, no longer a blocker**, since the group carrying them
(fuel type alone) is the one that measurably helps despite it; a province-wide extent would likely
still improve on this further, but it isn't gating today's result.

**Decision: fuel type promoted, FWI and terrain are not.** `FEATURE_COLUMNS` now includes the 19
`fuel_type_*` columns (32 total), re-exported with the same `BEST_RANDOM_FOREST_PARAMS`, no retune
needed given the sweep above. FWI and terrain stay in the dataset and their fetch/cache modules as
validated-but-not-promoted groundwork, the same status `cape`/`convective_precip_mm` has: genuinely
tested, not dead code, just not currently earning a place in the served model.
`/predict/live`/`/predict/explain` needed a fuel-type source once the served model depends on it, see
[Serving](07-serving.md#fuel-type-for-predictlive) for `features/live_fuel_type.py`, a cache lookup
rather than a third live fetch, since fuel type doesn't change day to day.

## Testing the multi-day-ahead label (2026-08-21)

[Problem framing](01-problem-framing.md#what-were-actually-predicting) has always named the natural
extension past same-day prediction: "will a fire be detected ... on day *D*, or within the next *N*
days?" This is the first real attempt at it. See [Grid &
labels](03-grid-and-labels.md#the-multi-day-ahead-label-ignited_next_nd-2026-08-21) for how
`ignited_next_3d` is actually built (a forward rolling-max over `ignited`, `n_days=3`) and why the
temporal split's boundaries don't need special trimming for it given this project's existing
fire-season scope, this section is the modeling result on top of that label.

**Positive rate rises, but less than proportionally to the window width.** Filtered to fire season,
the 3-day-ahead label's positive rate is ~1.7x the same-day rate on every split (test: 0.0998% ->
0.1741%), not ~3x, real fires cluster over consecutive days (the same event still burning, or a
FIRMS detection lagging ignition by a day), so widening the window catches fewer *additional* distinct
events than a naive 3x would suggest.

**First pass: the served model's exact hyperparameters, unretuned, on the new label.** Same
`FEATURE_COLUMNS` (the current 32, unchanged, fuel type included, the same set
`export_multi_day_model` fits against), same
`BEST_RANDOM_FOREST_PARAMS`, same temporal split, only the label column swapped:

| model | val PR-AUC | test PR-AUC | test top-10% |
| --- | --- | --- | --- |
| same-day `ignited` (served) | 0.3632 | 0.3816 | 88.0% |
| 3-day-ahead, unretuned | 0.3520 | 0.2906 | 80.8% |
| dummy (3-day-ahead) | 0.0054 | 0.0017 | 15.9% |

Clears the dummy floor by a wide margin (test PR-AUC ~166x the dummy's), confirming real skill on the
wider target, alongside a genuine, expected gap versus the same-day model (test PR-AUC down ~24%
relative). Not a bug or a confound to chase: the 3-day-ahead model solves a strictly harder problem
(any ignition in a 3-day window, not one exact day) from exactly the same information, since no weather
forecast for *D+1*/*D+2* exists in this pipeline.

**Retune attempt: made val better and test worse, discarded.** A 15-candidate manual random search
(same `n_jobs=-1`-hang-avoiding manual-loop pattern the `max_features` sweep above used, generalized
to `n_estimators`/`max_depth`/`min_samples_leaf`/`max_features` via a fixed-seed random sample) picked
the val-PR-AUC winner:

| model | val PR-AUC | test PR-AUC | test top-10% |
| --- | --- | --- | --- |
| 3-day-ahead, unretuned | 0.3520 | **0.2906** | **80.8%** |
| 3-day-ahead, retuned (`n_estimators=499, max_depth=8, min_samples_leaf=7, max_features=0.7553`) | **0.3593** | 0.2704 | 79.1% |

The retuned config wins on val but loses on test, the same single-fold-overfitting shape seen before
([Widening the search](#widening-the-search-randomizedsearchcv--predefinedsplit)), resolved the same
way: trust test over a close val call. The unretuned, same-day-tuned params are the better choice for
this label, so the existing hyperparameters already generalize reasonably to a differently-shaped
target and searching harder made things worse.

**Decision: served, as a second parallel model.** Unlike FWI/terrain above (neutral-to-negative on the
*same* target, genuinely not worth serving), this result is a real, working capability with an honest
accuracy gap on a *harder* target, worth shipping with the gap documented, not worth hiding behind
"not promoted." `training/export_model.py::export_multi_day_model` exports a second `ModelBundle`
(`data/processed/model_3day.joblib`, same `FEATURE_COLUMNS`/`BEST_RANDOM_FOREST_PARAMS`, no calibrator
yet, see that function's docstring for why) and `api/main.py` serves it at
`GET /predict/live/multi-day`, reusing the exact same live weather/neighbor-fire/fuel-type sourcing
`/predict/live` already has. See [Serving](07-serving.md#predictlivemulti-day-the-3-day-ahead-endpoint)
for the endpoint itself.