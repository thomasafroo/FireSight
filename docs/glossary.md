# Glossary

Every term used across the other pages, defined once here and linked back to from wherever it's
used.

### Brier score
The mean squared error between a predicted probability and the actual {0, 1} outcome, averaged over
every row. 0 is perfect. Unlike PR-AUC/ROC-AUC/top-k% capture, which only care about *ranking*, Brier
score is sensitive to whether a probability is right in absolute terms, so it's the metric that
catches a model whose ordering is fine but whose numbers are meaningless. The informative comparison
isn't the raw value but its ratio against a **base-rate-only floor**, `p * (1 - p)` for a base rate
`p`, which is what a model scores by ignoring every feature and always predicting the true positive
rate. This project's served model scored 17-489x *worse* than that floor before a calibrator was
attached. See [calibration](#calibration-isotonic-and-platt-scaling) and [Modeling &
evaluation](06-modeling-and-evaluation.md#calibration-is-ignition_probability-a-real-probability).

### Calibration (isotonic and Platt scaling)
A model is **calibrated** if, among all the rows it scores at 0.7, about 70% really are positive.
Ranking metrics can't see this at all, so a model can top every metric on this project's list while
its raw `predict_proba` output is off by two orders of magnitude, which is exactly what happened here:
`class_weight="balanced"` deliberately trains against a rebalanced class distribution, so the
probabilities that fall out describe that rebalanced world, not the real 0.1-0.3% base rate.
**Recalibration** fixes the scale after the fact by fitting a second, monotonic model from raw score
to observed frequency. **Isotonic regression** learns a flexible step-shaped curve; **Platt scaling**
(also called sigmoid calibration) fits a single logistic curve instead. Because either one is
monotonic, it changes magnitudes without changing order, so ranking is unaffected.
`evaluation/calibration.py` implements both, and the served bundle carries a pooled isotonic one. See
[Brier score](#brier-score), [Modeling &
evaluation](06-modeling-and-evaluation.md#does-pooled-leave-one-year-out-validated-calibration-actually-help),
and [Serving](07-serving.md#calibration-ignition_probability-vs-calibrated_probability).

### CAPE (convective available potential energy)
A meteorological measure, in joules per kilogram, of how much energy a rising parcel of air would
release if it kept rising, i.e. how much the atmosphere is offering a thunderstorm updraft. High CAPE
means conditions favour convective storms, and therefore lightning, the ignition source behind 59.5%
of this region's fire-season fires. It's used here as a *proxy*: no public lightning-strike feed
covers this project's 2012-2024 training years, but CAPE does, and it comes from the same ECMWF
reanalysis family as the rest of the weather. Fetched from full ERA5 (~0.25° grid) rather than
ERA5-Land (which doesn't carry it at all), and aggregated daily as a **max** rather than a mean, since
an afternoon instability spike is the relevant number and an average washes it out. Joined into the
dataset but not currently in `FEATURE_COLUMNS`. See [Weather
join](04-weather-join.md#a-second-weather-source-full-era5s-cape-and-convective-precipitation) and
[Modeling &
evaluation](06-modeling-and-evaluation.md#adding-capeconvective_precip_mm-to-feature_columns).

### Circular encoding
A way of representing an angle (compass direction, hour of day, day of year) as two numeric columns,
`sin(θ)` and `cos(θ)`, instead of one raw number. A raw angle column falsely treats 359° and 1° as
far apart (358 units) when they're actually 2° apart; sin/cos preserves true circular distance. See
[Feature engineering](05-feature-engineering.md#what-actually-got-built) (`wind_dir_sin`/
`wind_dir_cos` are still computed there, though they were dropped from the served model's
`FEATURE_COLUMNS`, see
[Modeling & evaluation](06-modeling-and-evaluation.md#dropping-the-dead-weight-features)).

### Class imbalance
When one class (here, `ignited == 1`) makes up a small fraction of the data, 0.16% in this dataset.
Standard techniques (accuracy, unweighted loss functions) implicitly assume roughly balanced classes
and behave badly under imbalance, which is why this project uses PR-AUC/top-k instead of accuracy,
and `class_weight="balanced"` in `LogisticRegression`. See
[Problem framing](01-problem-framing.md#why-this-is-a-rare-event-classification-problem).

### Cross join
A join that pairs every row of table A with every row of table B, `len(A) x len(B)` rows out, no
matching key required. Used in `build_label_scaffold` to produce every `(cell_id, date)` combination
before labeling which ones had a fire. See
[Grid & labels](03-grid-and-labels.md#why-the-label-table-needs-a-full-cross-product-not-just-fire-rows).

### ERA5-Land
See [reanalysis](#reanalysis) and
[Data sources](02-data-sources.md#era5-land-weather-the-feature-source).

### FBP System (Canadian Forest Fire Behaviour Prediction System)
The Canadian national system for predicting how a fire will *behave* once it's burning (rate of
spread, intensity, fuel consumed), as distinct from the [FWI
System](#fwi-system-canadian-forest-fire-weather-index-system), which rates how dangerous the
*weather* is. FBP takes three input categories: fuel type, weather, and terrain. Its fuel-type
classification is the vocabulary BC's Provincial Fuel Type Layer is written in: `C-1`..`C-7` conifer,
`D-1`/`D-2` deciduous, `M-1`..`M-4` mixedwood, `S-1`..`S-3` slash, `O-1a`/`O-1b` grass, `N` non-fuel,
`W` water. FireSight doesn't model fire behaviour, but it does borrow that vocabulary as a categorical
feature: 19 codes occur in the Kamloops extract and are one-hot encoded into `FEATURE_COLUMNS`. See
[Data sources](02-data-sources.md#bc-provincial-fuel-type-layer-fuel-type-a-feature-source) and
[Modeling &
evaluation](06-modeling-and-evaluation.md#per-group-ablation-isolating-which-of-the-three-actually-helps).

### FIRMS
NASA's **Fire Information for Resource Management System**, the source of fire *detection* points
used as labels in this project. See
[Data sources](02-data-sources.md#firms-fire-detections-the-label-source).

### FWI System (Canadian Forest Fire Weather Index System)
The fire-danger rating system BC Wildfire Service runs operationally: six numbers computed
**recursively** day over day from temperature, relative humidity, wind, and 24-hour rain, so each
day's value depends on the previous day's. Three are fuel-moisture codes on different timescales,
**FFMC** (fine fuels like surface litter, responds within hours), **DMC** (loosely compacted duff,
days), and **DC** (deep compacted organic layers, weeks to months, the drought memory). Three are
behaviour indices built from those: **ISI** (FFMC plus wind, expected spread rate), **BUI** (DMC plus
DC, how much fuel is available to burn), and **FWI** itself (ISI plus BUI, overall intensity).
`features/fwi.py` implements the Van Wagner equations from data already in the pipeline, no new
source, and resets the recursion to standard start-up values on a fixed March 1 each year, since
there's no snow-cover data to derive a real spring-melt date from. Computed and stored, but measured
as neutral and not promoted into the served model. See [Modeling &
evaluation](06-modeling-and-evaluation.md#closing-the-feature-category-gap-fwi-terrain-and-fuel-type-2026-08-21).

### Hyperparameter / grid search
A **hyperparameter** is a setting you choose *before* fitting a model (e.g. `RandomForest`'s
`max_depth`, `XGBoost`'s `learning_rate`), as opposed to a *parameter*, which the model learns from
data during `.fit()` (e.g. a linear model's coefficients). **Grid search** tries every combination
from a set of candidate values for each hyperparameter and keeps whichever combination scores best
on validation data. `training/advanced_models.py::tune_model` does this manually in a loop against
the fixed 2023 temporal validation split, rather than using `sklearn.model_selection.GridSearchCV`,
`GridSearchCV` does its own internal cross-validation (randomly splitting the training data into
folds), which would reintroduce exactly the random-split temporal leakage `temporal_split` exists to
avoid (see [leakage](#leakage-temporal--data-leakage)). This only rules out `GridSearchCV`'s *default*
cross-validation, though, see [PredefinedSplit](#predefinedsplit) for how the project later reused
`RandomizedSearchCV` safely by handing it the same fixed split instead. See
[Modeling & evaluation](06-modeling-and-evaluation.md#randomforest-and-xgboost-trainingadvanced_modelspy).

### Leakage (temporal / data leakage)
When information that wouldn't actually be available at prediction time ends up influencing training
or evaluation, making a model look better than it would perform in reality. Here, the specific risk
is *temporal* leakage: random splitting would let the model be evaluated on rows nearly identical
(spatially/temporally) to ones it trained on. See
[Modeling & evaluation](06-modeling-and-evaluation.md#splitting-by-time-never-randomly).

### MAP_KEY
The API key FIRMS requires for programmatic access (free, registered per-user). Set as
`FIRMS_MAP_KEY` in `.env`.

### ModelBundle
`training/persist.py::ModelBundle`, a small dataclass that bundles a fitted model together with the
exact `feature_columns` list (in order) it was trained on, plus free-form `metadata` (model type,
validation scores, when it was trained), `joblib`-serialized as one object. Exists so serving code
can never independently hardcode or guess the feature list and have it silently drift from what the
model actually expects. See [Serving](07-serving.md#persisting-a-model-without-losing-its-contract).

### MODIS
One of two satellite instrument families FIRMS distributes detections from (the other is
[VIIRS](#viirs)), coarser resolution (~1km), longer historical record (~2000-present). See
[Data sources](02-data-sources.md#firms-fire-detections-the-label-source).

### Moore neighborhood
On a square grid, the 8 cells touching a given cell, diagonals included, as opposed to the **von
Neumann** neighborhood, which counts only the 4 sharing an edge. FireSight's
`neighbor_fire_count_{1,3,7}d` features count how many of a cell's 8 Moore neighbors ignited in the
trailing N days, and the diagonals are included deliberately: a fire spreading southeast doesn't care
which of its neighbors happen to share an edge. Because `cell_id` is literally `"{row}_{col}"`,
finding the 8 neighbors is plain string arithmetic (`features/grid.py::neighbor_cell_ids`), with no
spatial index needed. See [Feature engineering](05-feature-engineering.md#what-actually-got-built) and
[Modeling &
evaluation](06-modeling-and-evaluation.md#spatial-lag-features-implemented-and-promoted-2026-08-19).

### Nearest-neighbor join
Matching each point in dataset A to whichever point in dataset B is spatially closest, rather than
requiring an exact coordinate match. Used to attach ERA5-Land's coarser weather grid to the finer
fire-detection grid. See [Weather join](04-weather-join.md#problem-1-two-different-grids).

### Panel data
Data with many distinct units (here, grid cells), each observed repeatedly over time, as opposed to
a single time series (one unit over time) or plain cross-sectional data (many units at one point in
time). Feature engineering here (lags, rolling windows) must be grouped by unit (`cell_id`) to stay
valid. See
[Feature engineering](05-feature-engineering.md#this-is-panel-data-not-a-single-time-series).

### PR-AUC (average precision)
Area under the precision-recall curve. Precision = of everything predicted positive, what fraction
really was; recall = of everything truly positive, what fraction got predicted. PR-AUC summarizes
the tradeoff across all thresholds into one number, and, unlike ROC-AUC, stays sensitive to model
quality even when negatives vastly outnumber positives, which is why it's the primary metric here.
See [Problem framing](01-problem-framing.md#the-metrics-used-instead).

### PredefinedSplit
An `sklearn.model_selection` cross-validation splitter that takes a fixed fold assignment instead of
computing one, you hand it a `test_fold` array (`-1` for rows that should always stay in training,
a fold index for rows that should be held out) and it reproduces exactly that split, no shuffling.
Used to let `RandomizedSearchCV`/`GridSearchCV` search hyperparameters against the project's existing
train/val split without their default cross-validation reshuffling the data and reintroducing the
same [leakage](#leakage-temporal--data-leakage) `temporal_split` exists to prevent. See
[Modeling & evaluation](06-modeling-and-evaluation.md#widening-the-search-randomizedsearchcv--predefinedsplit).

### Reanalysis
A dataset produced by running a physics-based weather/climate model *backward* over historical time,
constrained by whatever real observations exist, to produce a physically consistent, gap-free
estimate of conditions everywhere and every timestep, not raw observations, and not a forecast.
ERA5-Land is ECMWF's reanalysis product. See
[Data sources](02-data-sources.md#era5-land-weather-the-feature-source).

### ROC-AUC
Area under the ROC curve (true positive rate vs false positive rate across thresholds). Reported as
secondary context in this project, its false-positive-rate axis gets diluted by the huge number of
true negatives under class imbalance, so it can look deceptively good even when precision is poor.
See [Problem framing](01-problem-framing.md#the-metrics-used-instead).

### Rolling-origin backtest
Evaluating a model across *many* held-out time periods instead of one, by repeatedly moving the
train/test boundary forward: train on everything through year N-1, score year N, then advance and
repeat. The training window **expands** each fold rather than sliding, so every fold sees all the
history a real deployment would have had at that point. It answers a question a single test set
can't: is the reported number typical, or did it land on a lucky year?
`evaluation/backtest.py::run_rolling_origin_backtest` runs 8 folds (holdout years 2017-2024, with
2012-2016 reserved as a floor of history before the first) with hyperparameters held fixed, so the
evaluation year is the only thing that varies. Running it is what revealed that this project's
headline test-set score sat near the *best* year observed rather than a typical one. See [Modeling &
evaluation](06-modeling-and-evaluation.md#rolling-origin-backtest-is-719-typical-or-the-best-year-in-the-dataset).

### scale_pos_weight
`XGBClassifier`'s way of handling [class imbalance](#class-imbalance), `XGBoost` has no
`class_weight="balanced"` option like `LogisticRegression`/`RandomForestClassifier`, so
`training/advanced_models.py::fit_xgboost` computes the equivalent manually:
`scale_pos_weight = (count of negatives) / (count of positives)`, **recomputed from the training
fold only** each time a model is fit (never hardcoded, and never computed from validation/test), so
the correction can't leak information about val/test's class balance into training. See
[Modeling & evaluation](06-modeling-and-evaluation.md#randomforest-and-xgboost-trainingadvanced_modelspy).

### SHAP (SHapley Additive exPlanations)
A method for explaining *one* prediction rather than a model as a whole: it splits a single score
into one signed contribution per feature, and those contributions plus a base value sum back to the
score exactly. That additivity is the point, it's what lets you say "this cell was flagged because
`neighbor_fire_count_7d` added +0.52," where the global [feature-importance
measures](06-modeling-and-evaluation.md#feature-importance-what-the-model-is-actually-leaning-on) can
only say "soil moisture matters most on average." The name comes from Shapley values in cooperative
game theory, which SHAP borrows to divide credit among features fairly. `shap.TreeExplainer` computes
these in milliseconds for a tree ensemble, which is why `/predict/explain` can serve them live. One
limit worth knowing: it decomposes the raw `ignition_probability` only, since the isotonic calibrator
sits outside the tree structure it reads. See [Modeling &
evaluation](06-modeling-and-evaluation.md#shap-explainability-implemented-2026-08-19) and
[Serving](07-serving.md#explaining-a-live-prediction-predictexplain).

### Top-k% capture
Of all actual positives, the fraction that fall within the model's top-k%-scored predictions. Framed
operationally: if only the top 10% riskiest cell-days could be acted on (limited crews/resources),
what fraction of real fires would that have caught? See
[Problem framing](01-problem-framing.md#the-metrics-used-instead).

### Venn-Abers
A calibration method that returns, per prediction, not just a corrected probability but an
**interval** `[p0, p1]` around it, a per-row statement of how far that number can be trusted rather
than one global caveat in the docs. Tested here against the same 8-fold leave-one-year-out rig the
isotonic/sigmoid calibrators went through, with the success criterion written down first: intervals
had to widen in the years already known to be least reliable. They didn't (the correlation came out
weakly *negative*), so it was recorded as a clean negative result and nothing was shipped. One
implementation trap worth keeping: the library's high-level `VennAbersCalibrator` wrapper does its own
**random** internal split, which would reintroduce exactly the
[leakage](#leakage-temporal--data-leakage) this project's temporal discipline exists to prevent. Only
the low-level `VennAbers` class, which takes a pre-computed calibration set, is safe here. See
[Modeling &
evaluation](06-modeling-and-evaluation.md#3-venn-abers-per-prediction-uncertainty-not-generic-conformal-prediction).

### VIIRS
The satellite instrument family (on Suomi NPP / NOAA-20) this project's FIRMS source
(`VIIRS_SNPP_SP`) uses, higher spatial resolution (~375m) than MODIS, detections since ~2012. See
[Data sources](02-data-sources.md#firms-fire-detections-the-label-source).

### WFS (Web Feature Service)
An OGC standard for serving *vector* geographic data (points, lines, polygons, with their attributes)
over HTTP, where a client asks for the features matching a query instead of downloading the whole
dataset. That distinction is load-bearing here twice over: BC's Provincial Fuel Type Layer is a ~4GB
File Geodatabase needing GDAL to read as a download, but a per-cell bounding-box query away over WFS,
and BC Wildfire Service's historical incident records were queried the same way to confirm the winter
blind spot's human-cause explanation. Together they're how FireSight uses real provincial geospatial
data without taking on a geo stack. See [Data
sources](02-data-sources.md#bc-provincial-fuel-type-layer-fuel-type-a-feature-source) and [Modeling &
evaluation](06-modeling-and-evaluation.md#known-limitation-a-wintershoulder-season-blind-spot).
