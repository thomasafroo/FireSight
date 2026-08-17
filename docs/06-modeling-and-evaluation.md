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

A plain `Pipeline`, not a `ColumnTransformer`, in the end — see below for why. A few things worth
being explicit about:

- **No `ColumnTransformer` needed at all, in the end.** Every feature that made it into
`FEATURE_COLUMNS` is numeric (temperature, soil moisture, precipitation, wind speed,
days-since-rain, etc.) — there's no categorical column to route separately, so a single
`StandardScaler` applied to everything is enough; `ColumnTransformer` only earns its keep once
different columns need different treatment. `cell_id` is deliberately *not* fed in as a raw feature
— treating it as a categorical column (even one-hot encoded) would let the model partly memorize
"this specific cell tends to burn," which can't generalize to a cell it hasn't seen enough fire
history for, and muddies the temporal-generalization story the whole train/val/test split is
designed to test honestly.
- **Scaling only matters for the linear model.** `StandardScaler` is needed for `LogisticRegression`
(gradient-based optimization converges better and regularization behaves sanely when features are on
comparable scales) but is a no-op for tree-based models (`RandomForestClassifier`, `XGBoost`) —
trees split on thresholds per feature independently, so the scale of a feature doesn't change what
splits are chosen. The pipeline can stay a no-scaling passthrough for tree models, or keep the
scaler in place harmlessly.
- **Fit only on train.** `StandardScaler` is `.fit()` on the train split only, then `.transform()`
on val and test — fitting it on the full dataset (including val/test) before splitting would leak
information about their distribution into a preprocessing decision, a subtler version of the same
leakage problem the temporal split exists to prevent. (The rolling-feature warm-up `NaN`s are a
separate, earlier step: rows are dropped outright — not imputed — in `pipeline/build_dataset.py`
before the split even happens, since a dropped row has no split-dependent behavior to leak; see
[Feature engineering](05-feature-engineering.md#handling-the-nans-this-introduces).)

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
41.9%) is the same directional pattern seen before the scope change, just more pronounced. See
[Investigating the val/test gap](#investigating-the-valtest-gap-a-monthly-breakdown) below for what
actually explains it.

## Investigating the val/test gap: a monthly breakdown

The re-tune above widened the val/test gap (test top-10% 71.9% vs. val 41.9%) enough to be worth
checking with the same tool that originally surfaced the winter blind spot: a monthly breakdown of
which fires the served model's own top-10%-by-risk cutoff catches vs. misses, run separately on val
(2023) and test (2024) under the current fire-season scoping.

| month | val (2023) caught/total | val capture | test (2024) caught/total | test capture |
|---|---|---|---|---|
| May | 0/4 | 0% | 0/6 | 0% |
| Jun | 0/6 | 0% | — | — |
| Jul | 47/101 | 46.5% | 102/120 | 85.0% |
| Aug | 282/538 | 52.4% | 72/111 | 64.9% |
| Sep | 7/153 | 4.6% | — | — |
| Oct | — | — | 0/5 | 0% |

The gap isn't a val-vs-test modeling artifact — it's a **fire-count distribution difference between
the two specific years landing on a month the model is already comparatively weak at**. Val (2023)
had 153 fires in September alone (19% of its 802 total) that the model catches only 4.6% of the time
(mean predicted risk on those fire-days: 0.31); test (2024) had essentially no September fires (0 out
of 242) to be dragged down by. July/August capture rates are actually *higher* on val than the
aggregate 41.9% suggests (46.5%/52.4%), much closer to test's July/August numbers (85.0%/64.9%) —
it's the September cluster alone pulling val's blended number down, not systematically weaker
July/August performance. This lines up with 2023 being BC's worst fire season on record, with
large fires still active into September, while 2024 was comparatively quiet outside of a concentrated
July/August peak.

**Why September specifically is weak, structurally:** the same reasoning as the [winter/shoulder-
season blind spot](#known-limitation-a-wintershoulder-season-blind-spot) applies in miniature.
September sits at the tail of the fire-season window — cooler and wetter on average than peak
July/August — so it's a smaller-scale version of the same "hot+dry = risk" blind spot the model has
for fully excluded winter months, just not severe enough on its own to justify excluding September
from the fire-season window entirely (doing so would also throw away 153 real September positives
from *training* data in every year, not just val's). Treated as the same documented weather-only
limitation as the winter blind spot, not a bug — no code change made here, since the gap is explained
by real between-year variation in *when* fires happened, not by a leak, split-boundary bug, or
scoring inconsistency between val and test.

## Rolling-origin backtest: is 71.9% typical, or the best year in the dataset?

The investigation above explains the *val-vs-test* gap, but it only compares two years. The deeper
question it raises — "how much does the reported number move if you happen to test on a different
year?" — needs more than two data points to answer honestly. `evaluation/backtest.py` answers it: keep
`BEST_RANDOM_FOREST_PARAMS` fixed (no re-tuning per fold, to isolate "does the *evaluation year* matter"
from "would retuning help"), refit on an **expanding** training window (2012 through year N-1), and
score against each subsequent year N in turn, for N = 2017..2024 (2012-2016 reserved as a five-year
floor before the first holdout, so the earliest fold isn't evaluating off a single year of history).

| year | train rows | holdout positives | PR-AUC | ROC-AUC | top-10% capture |
|---|---|---|---|---|---|
| 2017 | 1,212,120 | 1,433 | 0.0141 | 0.763 | 30.8% |
| 2018 | 1,454,544 | 229 | 0.0023 | 0.762 | 25.3% |
| 2019 | 1,696,968 | 57 | 0.0004 | 0.623 | 22.8% |
| 2020 | 1,939,392 | 43 | 0.0003 | 0.653 | 16.3% |
| 2021 | 2,181,816 | 2,628 | 0.0333 | 0.829 | 27.0% |
| 2022 | 2,424,240 | 98 | 0.0005 | 0.632 | **8.2%** |
| 2023 | 2,666,664 | 802 | 0.0136 | 0.833 | 41.8% |
| 2024 | 2,909,088 | 242 | 0.0113 | 0.884 | **74.4%** |

**Sanity check first:** the 2023 row (train on everything before 2023) reproduces the original `val`
numbers from [Re-tuning after the fire-season scope
change](#re-tuning-after-the-fire-season-scope-change) *exactly* (PR-AUC 0.0136, ROC-AUC 0.833,
top-10% 41.8%) — expected, since it's the same train set and the same holdout year, just computed by
this script instead of `export_model.py`. The 2024 row is close to but not identical to the originally
reported `test` numbers (71.9% there vs. 74.4% here) for a real reason, not noise: `export_model.py`'s
served model trains only through 2022 (`TRAIN_END = "2023-01-01"`) and is scored against *both* val
2023 and test 2024 without ever training on 2023's data, while this fold's training window includes
2023 (one extra year) before scoring 2024 — a small, expected boost from more training data, not a
discrepancy to chase.

**The headline number turns out to be close to the best year observed, not a typical one.** Across all
8 folds: top-10% capture has mean **30.8%**, median **26.2%**, and a standard deviation of **~19
percentage points** — on a metric that's bounded at 0% and 100%, that's enormous relative spread. The
71.9%/74.4% number this project has been citing as "the" result sits at the *top* of this range, tied
with 2023 as the two best years out of eight; five of the eight backtested years land at 30% or below,
and 2022 — trained on *more* data than 2017 or 2021 — is the worst of all at 8.2%.

**What doesn't explain the swing:** more training data. Training rows grow monotonically from 1.2M
(2017) to 2.9M (2024), but top-10% capture doesn't track that at all (2022, with double 2017's
training data, scores a quarter of 2017's number). Nor does raw fire count in the holdout year: 2017
and 2021 (BC's two worst fire seasons on record, 1,433 and 2,628 positives) score *worse* than 2023's
802-positive year and far worse than 2024's comparatively quiet 242-positive year. **What's more
consistent with the evidence:** the same structural weakness documented in [Investigating the val/test
gap](#investigating-the-valtest-gap-a-monthly-breakdown) and [Known
limitation](#known-limitation-a-wintershoulder-season-blind-spot) above — the model is much stronger in
peak July/August than at the fire-season's shoulders — so a year's score depends heavily on *when
within the season* its fires happened to land, which swamps both "how many fires" and "how much
training data" as an explanation. See [Why performance swings by month and
year](#why-performance-swings-by-month-and-year) below for the month-by-month breakdown that actually
tests this hypothesis, rather than just ruling out the two simpler explanations.

**One reassuring, consistent result across every fold:** ROC-AUC never dropped below 0.62 in any of
the 8 years (mean 0.747), meaning the model beat random ranking (0.5) in every single year tested,
including the worst ones. The instability is in *how much* better than chance the model is in a given
year, not *whether* it's better than chance at all — the model has real, year-round signal, it's just
unevenly distributed across the fire-season calendar.

**Practical takeaway:** don't repeat "71.9% top-10% capture" as if it were the model's expected
real-world performance — cite the full range (8-74%, median ~26%) or, if a single number is needed,
the median rather than the best-observed year. This doesn't undo the earlier finding that the model
beats chance by a wide, real margin (see [Baseline
results](#baseline-results-2023-validation-set)) — it means the size of that margin is much less
certain, and much more year-dependent, than a single test-set number could show.

## Why performance swings by month and year

The [backtest above](#rolling-origin-backtest-is-719-typical-or-the-best-year-in-the-dataset)'s leading
hypothesis — that a year's score depends on *when in the season* its fires land — was only tested
informally on two years so far ([Investigating the val/test
gap](#investigating-the-valtest-gap-a-monthly-breakdown)). `evaluation/backtest.py::
monthly_capture_breakdown` generalizes that by-hand table to every one of the 8 rolling-origin folds,
using the exact same top-10%-by-risk cutoff `top_10pct_capture` scores against, broken down by month.

**Pooled across all 8 years, the month-level pattern is unambiguous and consistent:**

| month | fires (8 years pooled) | caught | capture rate |
|---|---|---|---|
| May | 53 | 2 | **3.8%** |
| Jun | 45 | 19 | 42.2% |
| Jul | 2,289 | 903 | 39.4% |
| Aug | 2,522 | 712 | 28.2% |
| Sep | 409 | 90 | 22.0% |
| Oct | 214 | 27 | **12.6%** |

May is by far the weakest month in the entire fire-season window — worse, in relative terms, than the
already-documented September weakness — despite being included as "in season." October is the second
weakest. June and July are the strongest, and August, despite being colloquially "peak fire season," is
meaningfully weaker than July (28.2% vs. 39.4%) once pooled across 8 years rather than read off a single
year. This is the clearest confirmation yet that the model's skill genuinely is concentrated in the
core of the season and thins out at both edges — not just a two-year coincidence.

**But month mix only partly explains the year-to-year swing, and the residual is informative.** Ranking
each year by the *share* of its fires that landed in Jul/Aug (the two strongest months) against its
overall `top_10pct_capture` gives a correlation of **r = 0.63** across the 8 folds — real and positive,
but far from a complete explanation:

| year | % of that year's fires in Jul/Aug | top-10% capture |
|---|---|---|
| 2019 | 3.5% | 22.8% |
| 2020 | 32.6% | 16.3% |
| 2022 | 34.7% | 8.2% |
| 2018 | 39.3% | 25.3% |
| 2023 | 79.7% | 41.8% |
| 2017 | 87.4% | **30.8%** |
| 2021 | 97.0% | **27.0%** |
| 2024 | 95.5% | 74.4% |

**2017 and 2021 are the outliers that keep this from being a clean story.** Both had 87-97% of their
fires in the nominally-strongest months, yet both scored only 27-31% — worse than 2023's 79.7%-in-
Jul/Aug year, and far worse than 2024's similarly Jul/Aug-heavy year (95.5% -> 74.4%). Both 2017 and
2021 are BC's two worst fire seasons on record, and both show the same specific pattern when broken
down further: 2021's August alone (1,009 fires — roughly 2-4x a typical year's *entire* season) was
caught only **11.7%** of the time (118/1,009), even though August is a nominally strong month in every
other year. That's consistent with a second, independent factor beyond month-of-season: **extreme,
high-volume fire seasons seem to break the ranking even within their strong months**, plausibly because
`top_10pct_capture` ranks a fixed ~24,000-row slice against the *entire* year at once — a season with
several times the normal fire count has that much more competition for the same fixed number of "top
10%" slots, and/or extreme seasons plausibly involve a higher share of fires whose spread is driven by
wind/fuel-continuity effects this weather-only feature set doesn't capture, not just heat and dryness.

**Follow-up investigation, confirmed both mechanisms above are real.** Refitting just the 2017 and 2021
folds (plus 2024 as a normal-year control) and comparing caught vs. missed fires directly on two axes —
how many *other* cells ignited that same day, and how the raw weather features differ — found two
separate, compounding effects, not one:

**1. Extreme same-day fire counts directly overwhelm the fixed top-10% budget, but only past a
threshold.** Bucketing each year's fires by how many cells ignited on that exact date:

| year | 1 fire/day | 2-5 | 6-20 | 21-50 | 51-100 | 100+ |
|---|---|---|---|---|---|---|
| 2024 (control) | 0% | 38.1% | 83.6% | 80.0% | — | — |
| 2017 | 7.7% | 20.6% | 33.6% | 31.4% | 23.1% | — |
| 2021 | 0.0% | 15.2% | 24.2% | **41.5%** | 28.2% | **15.1%** |

In the normal-year control (2024, max 30 same-day fires), capture *rises* with same-day fire count —
more simultaneous ignitions means a hotter/drier/windier day, exactly the pattern the model is tuned to
flag, so it catches most of a busy day at once. **2021 shows the opposite past a point**: capture peaks
at 41.5% for 21-50-fire days, then *falls* to 28.2% and then 15.1% as same-day counts climb past 50 and
past 100 — 2021 had entire days with 100+ simultaneous ignitions (581 fires fall in that single bucket
alone), something no other backtested year came close to. `top_10pct_capture` ranks a fixed ~24,000-row
slice against the *whole year* (~144 rows/day if spread evenly across the 168-day season) — a single
day with 100+ real fires mechanically cannot all fit in that day's share of the budget even if the
model correctly flags the whole day as extreme risk, especially when neighboring extreme days are
competing for the same fixed slots. 2017 shows the same declining-past-the-peak shape at a smaller
scale (31.4% -> 23.1% from the 21-50 to 51-100 bucket) without ever reaching 2021's 100+ regime.

**2. The weather-based signal itself is less discriminative in extreme years.** Comparing mean feature
values between caught and missed fires:

| | 2024 (control): caught vs. missed | 2021: caught vs. missed |
|---|---|---|
| `t2m` | 295.2K vs. 289.9K (**5.3K gap**) | 294.6K vs. 293.1K (**1.5K gap**) |
| `relative_humidity` | 38.6% vs. 54.1% (**15.5pp gap**) | 34.6% vs. 43.3% (**8.7pp gap**) |
| `swvl1` | 0.171 vs. 0.263 | 0.165 vs. 0.181 |

The same "hot+dry gets caught, cool+wet gets missed" direction holds in every year, but the *gap*
between caught and missed is roughly a third the size in 2021 as in the 2024 control. That's consistent
with 2021's defining feature as a fire season: the June 2021 BC heat dome put much of the province
under extreme heat/drought *simultaneously*, which compresses exactly the kind of cell-to-cell weather
variation the model relies on to rank — when nearly every cell looks dangerously hot and dry at once,
there's less relative signal left to separate which specific ones actually ignite that day, on top of
the fixed-budget problem in (1).

**Together:** month-of-season explains most of the routine year-to-year swing (r=0.63 above), and these
two compounding effects — a fixed annual ranking budget breaking down on the most extreme same-day fire
counts, plus weather variation itself compressing during province-wide extreme events — explain why the
two most severe fire seasons specifically underperform relative to their Jul/Aug-heavy month mix. Both
are structural properties of a global, weather-only ranking approach on binary-outcome data, not bugs
to fix with more features — the same conclusion the winter/shoulder-season blind spot investigation
below reaches for a different structural gap.

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
2. **Wind and the 7-day temp trend are dead weight in this model** — see [Dropping the dead-weight
   features](#dropping-the-dead-weight-features) below for what was actually done about it.

## Dropping the dead-weight features

The table above was computed once, before the fire-season/2012 re-tune settled. Before actually
removing anything from `FEATURE_COLUMNS`, permutation importance was re-run from scratch against the
*currently served* RandomForest (30 repeats, two random seeds, val and test both) to confirm the
picture still held:

| feature | val ΔPR-AUC (seed 1) | val ΔPR-AUC (seed 2) | test ΔPR-AUC (seed 1) | test ΔPR-AUC (seed 2) |
|---|---|---|---|---|
| `wind_dir_cos` | -0.000068 | -0.000067 | -0.000072 | -0.000072 |
| `u10` | -0.000323 | -0.000203 | -0.000166 | -0.000076 |
| `v10` | +0.000004 | +0.000001 | +0.000065 | +0.000072 |
| `wind_dir_sin` | +0.000555 | +0.000561 | +0.000032 | +0.000052 |
| `d2m` | +0.000287 | +0.000173 | -0.000339 | -0.000220 |
| `t2m_trend_7d` | +0.000017 | +0.000006 | +0.000046 | +0.000067 |
| `wind_speed` | -0.000211 | -0.000092 | **+0.000443** | **+0.000655** |
| `rh_mean_7d` (kept, for scale) | +0.001161 | +0.001220 | +0.000841 | +0.000956 |

`wind_dir_cos`, `u10`, `v10`, `wind_dir_sin`, `d2m`, and `t2m_trend_7d` all confirmed near-zero or
mixed-sign on both splits, both seeds — none of them cross even 15% of `rh_mean_7d`'s (the weakest
*kept* feature's) importance on either split, and several are actively negative. These six were
dropped from `FEATURE_COLUMNS`.

**`wind_speed` was kept, deviating from the original table's grouping.** Fresh evidence shows a real,
reproducible positive signal on the test split specifically — +0.00044 to +0.00066 across both seeds,
roughly half of `rh_mean_7d`'s test importance and nowhere near the ~0 cluster the other five sit in —
even though it's slightly negative on val. Test is the split this project's own methodology treats as
the honest, non-overfit read (see [Widening the
search](#widening-the-search-randomizedsearchcv--predefinedsplit) above), so this was trusted over the
weaker, noisier val signal. `u10`/`v10` (the raw wind vector components) were dropped in `wind_speed`'s
place instead — not originally singled out by name, but grouped with the same dead-weight bucket in
the table above, and confirmed consistently negative-or-negligible on both splits/seeds in the fresh
run, unlike `wind_speed` itself.

Refitting `BEST_RANDOM_FOREST_PARAMS` unchanged on the resulting 10-column `FEATURE_COLUMNS` (`t2m`,
`swvl1`, `precip_mm`, `relative_humidity`, `wind_speed`, `days_since_rain`, `precip_7d`, `precip_30d`,
`t2m_mean_7d`, `rh_mean_7d`) left val/test scores within noise of the 16-column version:

| | val PR-AUC | val ROC-AUC | val top-10% | test PR-AUC | test ROC-AUC | test top-10% |
|---|---|---|---|---|---|---|
| 16 columns (before) | 0.0138 | 0.832 | 41.9% | 0.0106 | 0.884 | 71.9% |
| 10 columns (after) | 0.0136 | 0.833 | 41.8% | 0.0106 | 0.884 | 71.9% |

Confirms these columns really were dead weight rather than something the tuned hyperparameters
happened to lean on — dropping them cost nothing measurable. The hyperparameters themselves weren't
re-tuned against the smaller feature set (that would mean re-running `tune_random_search`, which risks
the `n_jobs=-1` `RandomizedSearchCV` hang noted as unresolved future work); this is a same-params
refit only, and `data/processed/model.joblib` was re-exported from it via `export_model.py`. Nothing in
`features/engineering.py` changed — `add_relative_humidity`/`add_wind_features` still need `d2m` and
`u10`/`v10` as *inputs* to derive `relative_humidity` and `wind_speed`, and `ENGINEERED_COLUMNS` still
computes the dropped columns too (harmless, just unused by the model now); only the model's own input
list shrank.

**Why this matters beyond a smaller feature list:** it explains, in addition to the reasoning in
[Known limitation](#known-limitation-a-wintershoulder-season-blind-spot) above, why the winter/shoulder
-season blind spot was so resistant to more weather features — the model's real levers are almost
entirely slow-moving fuel-dryness signals (soil moisture, precip, temperature) plus one immediate
condition (wind speed), and nothing in `FEATURE_COLUMNS` encodes anything about human activity, which
is the more likely driver of winter ignitions. It also directly simplified [live weather
fetching](07-serving.md#live-weather-for-predictlive) for `/predict/live`: Open-Meteo can supply
relative humidity and wind speed *directly*, so the live-weather path never needs to reconstruct them
from dewpoint or wind vector components the way the ERA5-Land ingestion pipeline does.

## Testing the sequence-modeling hypothesis

`research/neural-networks.md` argues against a neural network replacing the served RandomForest, but
names one question it doesn't rule out: does a model that sees the **raw** last-30-days weather
sequence per cell — instead of the hand-engineered rolling summaries above (`t2m_mean_7d`,
`precip_30d`, ...) — capture a nonlinear temporal *shape* those summaries flatten away?
`training/sequence_model.py` runs that experiment: a small 1D-CNN (two `Conv1d` layers, global
average pooling, a small dense head) over 5 raw daily channels
(`t2m`/`precip_mm`/`swvl1`/`relative_humidity`/`wind_speed`, the same quantities behind the current
non-rolling features), trained with `BCEWithLogitsLoss(pos_weight=...)` — the PyTorch equivalent of
`class_weight="balanced"` — under the exact same temporal train/val/test split as everything else on
this page. The RandomForest side of the comparison is refit with the same tuned
`BEST_RANDOM_FOREST_PARAMS` on the *exact same row subset* the CNN sees (a handful of rows lose their
30-day raw window to date gaps that don't affect the rolling features), so the comparison is
apples-to-apples on identical rows, not just similar ones.

Real result:

| model                      | split      | pr_auc  | roc_auc | top_10pct_capture |
| --------------------------- | ---------- | ------- | ------- | ------------------ |
| RandomForest (tuned, same rows) | val (2023)  | **0.0136** | **0.833** | **41.8%** |
| SequenceCNN                 | val (2023)  | 0.0117  | 0.787   | 37.9%              |
| RandomForest (tuned, same rows) | test (2024) | **0.0106** | 0.884   | **71.9%**          |
| SequenceCNN                 | test (2024) | 0.0062  | 0.883   | 68.6%              |

The RandomForest wins on both splits, on every metric except test ROC-AUC (a near-tie, 0.884 vs.
0.883) — most clearly on PR-AUC (roughly 1.2-1.7x higher) and top-10%-capture (4-5 points higher on
both splits). The raw-sequence CNN did **not** find temporal shape the rolling-window features were
missing; if anything, letting the model learn its own temporal summary from scratch, on a training
set with only ~4,800 positive examples, generalized worse than the fixed 7-/30-day windows already
being handed to a tree ensemble. This matches the general tabular-data literature
`research/neural-networks.md` cites (tree ensembles beating deep learning on small, mostly-numeric
tabular data) rather than being an exception to it — the one hypothesis that document left open is now
closed, with a real negative result rather than an assumption.

**Practical conclusion:** no change to the served model. `BEST_RANDOM_FOREST_PARAMS` (via
`export_model.py`) stays the pick; the rolling-window features in `features/engineering.py` stay the
right representation of weather history for this problem, not a simplification that's costing
accuracy.

## Calibration: is `ignition_probability` a real probability?

Every metric on this page so far — PR-AUC, ROC-AUC, `top_10pct_capture` — is **rank-only**: each one
asks "are actual fires scored higher than non-fires," and every one of them is mathematically
unchanged by any monotonic rescaling of the raw scores. A model can top all three while its raw
`predict_proba` output is wildly wrong in absolute terms, which matters a lot here specifically,
because `api/main.py`'s `/predict` and `/predict/live` both hand a caller `ignition_probability` as a
bare float with no caveat — an obvious reading is "this cell has a 70% chance of igniting today" when
it says 0.7. `evaluation/calibration.py` checks whether that reading is actually justified, via two
tools that measure the thing rank metrics can't: **Brier score** (mean squared error between predicted
probability and the {0,1} outcome — 0 is perfect, and a model that just always predicts the true base
rate scores `base_rate * (1 - base_rate)`, a cheap floor to compare against) and a **reliability
table** (bucket predictions by predicted-probability quantile, then compare each bucket's mean
predicted probability against its actual observed fire rate — they should track each other for a
calibrated model).

Run against the served model on both held-out splits:

| | brier score | base-rate-only floor | observed positive rate |
|---|---|---|---|
| val (2023) | 0.1233 | 0.0033 | 0.33% |
| test (2024) | 0.0993 | 0.0010 | 0.10% |

**The served model's Brier score is ~40-100x *worse* (higher) than a trivial model that ignores every
feature and always predicts the split's true base rate.** That's a real, specific finding, not a
rounding effect — the reliability table shows exactly why:

| val (2023) predicted-probability bin | mean predicted | observed rate |
|---|---|---|
| 0.034 - 0.046 | 0.044 | 0.004% |
| 0.046 - 0.057 | 0.051 | 0.008% |
| 0.057 - 0.078 | 0.067 | 0.037% |
| 0.078 - 0.098 | 0.089 | 0.037% |
| 0.098 - 0.127 | 0.111 | 0.050% |
| 0.127 - 0.175 | 0.150 | 0.144% |
| 0.175 - 0.264 | 0.215 | 0.268% |
| 0.264 - 0.430 | 0.337 | 0.458% |
| 0.430 - 0.685 | 0.539 | 0.920% |
| 0.685 - 0.912 (top decile) | 0.852 | 1.382% |

The top decile — cells the model scores at a mean 85% ignition probability — actually ignites 1.38% of
the time on val (0.72% on test's equivalent bucket). Every bucket is monotonically ordered correctly
(higher predicted score really does mean higher observed rate, which is exactly why the *rank* metrics
above look good), but the absolute scale is off by roughly two orders of magnitude across the board,
worst at the high end where it matters most for anyone reading the number literally.

**Why, mechanically:** both `fit_random_forest` and `fit_logistic_regression` use
`class_weight="balanced"` (see [Baseline-first methodology](#baseline-first-methodology) above) —
that's a deliberate, correct fix for the optimizer collapsing to "always predict majority class"
during *fitting*, but it works by upweighting minority-class samples, which pushes `predict_proba`'s
output toward the artificially-rebalanced distribution the trees were actually fit against rather
than the true ~0.1-0.3% base rate. This is a known, expected side effect of `class_weight="balanced"`
— not a bug, and not something the earlier feature-drop or hyperparameter re-tuning on this page
would have caught, since none of `pr_auc`/`roc_auc`/`top_10pct_capture` can see it.

**What this does and doesn't mean:** the served model's *ranking* is still real and still the thing
`/risk-map` and the top-10%-capture story above rely on — nothing on this page about relative risk
changes. What changes is that `ignition_probability` in `api/main.py`'s responses should be read as a
**relative risk score, not a literal probability** — "this cell is much higher-risk than that one,"
not "this cell has an N% chance of burning" — see the caveat added to
[07-serving.md](07-serving.md#calibration-what-ignition_probability-does-and-doesnt-mean). Fixing the
absolute scale (e.g. `sklearn.calibration.CalibratedClassifierCV` with isotonic or Platt scaling,
fit on val) is a legitimate follow-up, but it's a separate decision from this measurement — isotonic
calibration on a split with only ~800 positives risks overfitting the calibration curve itself, and
recalibrating would need its own held-out check rather than reusing val/test as both the calibration
fit and the evaluation set.

### Is the miscalibration itself stable across years?

The [rolling-origin backtest](#rolling-origin-backtest-is-719-typical-or-the-best-year-in-the-dataset)
above already showed `top_10pct_capture` swings wildly by holdout year. `evaluation/backtest.py` also
computes Brier score and a reliability table for each of the same 8 folds, which answers a question
the single val/test calibration numbers above can't: is the *miscalibration factor* at least a stable
correction to apply, even if ranking performance isn't?

| year | positives | brier score | base-rate floor | brier ratio | top-decile mean predicted | top-decile observed | top-decile ratio |
|---|---|---|---|---|---|---|---|
| 2017 | 1,433 | 0.2056 | 0.0059 | 35.0x | 0.862 | 1.82% | 47x |
| 2018 | 229 | 0.0900 | 0.0009 | 95.4x | 0.795 | 0.24% | 332x |
| 2019 | 57 | 0.0506 | 0.0002 | 215.2x | 0.585 | 0.05% | 1,091x |
| 2020 | 43 | 0.0868 | 0.0002 | 489.3x | 0.772 | 0.03% | 2,673x |
| 2021 | 2,628 | 0.1826 | 0.0107 | 17.0x | 0.888 | 2.93% | 30x |
| 2022 | 98 | 0.1388 | 0.0004 | 343.5x | 0.855 | 0.03% | 2,590x |
| 2023 | 802 | 0.1233 | 0.0033 | 37.4x | 0.852 | 1.38% | 62x |
| 2024 | 242 | 0.1048 | 0.0010 | 105.1x | 0.814 | 0.74% | 110x |

**No — it's not stable either, and arguably worse than the ranking metric.** The Brier ratio alone
spans 17x to 489x (a ~29x spread), and the top-decile ratio spans 30x to 2,673x (an ~88x spread) — a
single "divide the raw probability by 50" correction that looked right in one year would be off by
another order of magnitude in another.

**But this needs one honest caveat before treating it at face value:** each `reliability_table` bin
holds ~24,000 (cell, date) rows, but in a sparse fire year almost none of them are actual fires — the
top-decile *observed rate* in a year with only 43-98 total positives is being estimated from a
handful of real fires landing in that one bin, and one extra or missing fire swings that rate (and
therefore the ratio) enormously in percentage terms. The pattern in the table above is consistent with
that: the two years with by far
the most positives (2021's 2,628, 2017's 1,433) — where the top-decile observed rate is estimated from
enough real fires to be statistically meaningful — show the *smallest and most similar* ratios (30x,
47x), while the sparsest years (2019's 57, 2020's 43, 2022's 98) show the wildest and largest ones. The
Brier ratio (computed over the *whole* holdout year, not just one bin of it) is somewhat less exposed
to this but shows the same shape (17-37x for the two big-fire years vs. 95-489x for the sparse ones).

**Practical reading:** the most statistically trustworthy estimate of "how miscalibrated is this model,
really" comes from the years with the most fires to estimate an observed rate from — 2017/2021/2023 all
cluster in the **17-47x** range for the Brier ratio, which is a more defensible number to reason about
than the full 17-489x spread. But this doesn't rescue the case for recalibrating now: a correction
factor estimated mostly from three unusually severe fire years (2017, 2021, and 2023, the latter itself
close to 2017/2021 in severity) is exactly the kind of single-scenario overfit this whole investigation
exists to catch — there is no evidence yet that the *same* correction would hold in a below-average
year like 2019 or 2020, and the small-sample years are too noisy to confirm or rule that out either
way. **Recommendation: don't fit a single static recalibration yet.** If/when this gets revisited, pool
observed-vs-predicted data across many years (not just the high-confidence ones) before fitting, and
validate the resulting calibrator's stability across individual holdout years the same way this table
does — a calibrator that only gets checked in aggregate could hide the exact same year-to-year
instability the aggregate `ignition_probability` numbers already did.

### Does pooled, leave-one-year-out-validated calibration actually help?

`evaluation/calibration.py::leave_one_year_out_calibration_check` does exactly what the recommendation
above asks for: for each of the 8 rolling-origin years, fit a calibrator on every *other* year's pooled
`(y_score, y_true)` pairs, apply it to the held-out year, and compare against doing nothing — the honest
test of whether pooling generalizes to a year it never saw, not whether it fits the years it was trained
on. Two calibration methods are compared: isotonic regression (a flexible monotonic curve) and sigmoid/
Platt scaling (a single logistic curve) — see `fit_isotonic_calibrator`/`fit_sigmoid_calibrator`.

| year | positives | pooled calibration-fit positives | raw Brier | isotonic Brier | sigmoid Brier | raw top-decile ratio | isotonic top-decile ratio | sigmoid top-decile ratio |
|---|---|---|---|---|---|---|---|---|
| 2017 | 1,433 | 4,099 | 0.2056 | 0.0059 | 0.0059 | 47.3x | 0.67x | 0.87x |
| 2018 | 229 | 5,303 | 0.0900 | 0.0010 | 0.0010 | 332.3x | 6.0x | 6.0x |
| 2019 | 57 | 5,475 | 0.0506 | 0.0002 | 0.0002 | 1,090.8x | 15.7x | 14.2x |
| 2020 | 43 | 5,489 | 0.0868 | 0.0002 | 0.0002 | 2,673.4x | 45.9x | 46.2x |
| 2021 | 2,628 | 2,904 | 0.1826 | 0.0107 | 0.0107 | 30.3x | 0.54x | 0.37x |
| 2022 | 98 | 5,434 | 0.1388 | 0.0004 | 0.0004 | 2,589.7x | 51.1x | 59.9x |
| 2023 | 802 | 4,730 | 0.1233 | 0.0033 | 0.0033 | 61.6x | 1.27x | 1.25x |
| 2024 | 242 | 5,290 | 0.1048 | 0.0010 | 0.0010 | 109.6x | 1.96x | 2.08x |

(top-decile ratio = mean predicted / observed rate in the top-scored bucket, matching the table above —
1.0x is perfect; both above and below 1.0x are miscalibrated.)

**Pooled calibration is a real, substantial improvement — but it doesn't solve the underlying
instability, it just moves the whole cluster of numbers much closer to correct.** Two separate results,
not one:

1. **Brier score improves by 15–500x in every single year**, isotonic and sigmoid performing almost
   identically throughout (no meaningful reason to prefer one over the other here). This is largely
   mechanical, not a deep achievement: Brier score is dominated by the huge majority of true-negative
   rows, and *any* calibration that shrinks scores toward the true ~0.1–0.3% base rate collapses their
   squared error, almost regardless of whether the shrinkage is precisely right for that specific year.
2. **The top-decile ratio — the number that actually answers "is a highly-scored cell's probability
   meaningful" — improves in absolute terms in every year (worst case 2,673x → 51x) but the *relative
   spread between the best- and worst-calibrated held-out year barely changes*: ~89x (2,673/30 raw) vs.
   ~95x (51/0.54 isotonic).** Pooling shifts every year's number much closer to 1.0x, but it doesn't make
   the years agree with each other any better than before — the year-to-year instability this whole
   investigation set out to check for is still there, just rescaled to a smaller absolute range.

**The years that stay worst-calibrated after pooling are exactly the sparsest-fire years** (2018's 229,
2019's 57, 2020's 43, 2022's 98 positives — still 6x–51x off), while the years with the most fires to
estimate a top-decile rate from land closest to 1.0x (2017 sigmoid: 0.87x, 2021 isotonic: 0.54x, 2023
isotonic: 1.27x). This matches the same statistical-noise explanation already given above for the raw
numbers: a sparse year's *observed* top-decile rate is itself being estimated from a handful of real
fires, so no calibrator — however well pooled — can be checked precisely against that little ground
truth in a single held-out year.

**Revised practical recommendation:** a pooled calibrator is worth having as a materially better default
than the raw score — it cuts the worst-case absolute miscalibration by roughly 50x — but it is not
"solved calibration." Its own accuracy is still meaningfully year-dependent, particularly for low-fire
years, in a way this single leave-one-year-out check can't fully rule out getting worse on a future year
unlike any of the 8 tested.

**Promoted to the served model**, on that basis: `training/export_model.py::export_current_best` now
fits this same pooled isotonic calibrator (all 8 years combined, not held-one-out — the LOYO split
above exists to validate the method, not to leave a year out of the production fit) and attaches it to
the `ModelBundle`, exposed via `/predict`'s and `/predict/live`'s `calibrated_probability` and
`/risk-map`'s `calibrated_risk_probability` (see [Serving](07-serving.md#calibration-ignition_probability-vs-calibrated_probability)).
`ignition_probability`/`risk_probability` are unchanged and still the field to use for ranking — the
calibrator only rescales magnitude, and the caveat above (particularly unreliable in low-fire years)
still applies to the calibrated number, so treat it as a much-improved estimate, not an exact one.
