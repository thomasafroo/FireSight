# Glossary

Every term used across the other pages, defined once here and linked back to from wherever it's
used.

### Circular encoding
A way of representing an angle (compass direction, hour of day, day of year) as two numeric columns,
`sin(θ)` and `cos(θ)`, instead of one raw number. A raw angle column falsely treats 359° and 1° as
far apart (358 units) when they're actually 2° apart; sin/cos preserves true circular distance. See
[Feature engineering](05-feature-engineering.md#planned-features).

### Class imbalance
When one class (here, `ignited == 1`) makes up a small fraction of the data — 0.16% in this dataset.
Standard techniques (accuracy, unweighted loss functions) implicitly assume roughly balanced classes
and behave badly under imbalance, which is why this project uses PR-AUC/top-k instead of accuracy,
and `class_weight="balanced"` in `LogisticRegression`. See
[Problem framing](01-problem-framing.md#why-this-is-a-rare-event-classification-problem).

### Cross join
A join that pairs every row of table A with every row of table B — `len(A) x len(B)` rows out, no
matching key required. Used in `build_label_scaffold` to produce every `(cell_id, date)` combination
before labeling which ones had a fire. See
[Grid & labels](03-grid-and-labels.md#why-the-label-table-needs-a-full-cross-product-not-just-fire-rows).

### ERA5-Land
See [reanalysis](#reanalysis) and
[Data sources](02-data-sources.md#era5-land--weather-the-feature-source).

### FIRMS
NASA's **Fire Information for Resource Management System** — the source of fire *detection* points
used as labels in this project. See
[Data sources](02-data-sources.md#firms--fire-detections-the-label-source).

### Hyperparameter / grid search
A **hyperparameter** is a setting you choose *before* fitting a model (e.g. `RandomForest`'s
`max_depth`, `XGBoost`'s `learning_rate`) — as opposed to a *parameter*, which the model learns from
data during `.fit()` (e.g. a linear model's coefficients). **Grid search** tries every combination
from a set of candidate values for each hyperparameter and keeps whichever combination scores best
on validation data. `training/advanced_models.py::tune_model` does this manually in a loop against
the fixed 2023 temporal validation split, rather than using `sklearn.model_selection.GridSearchCV` —
`GridSearchCV` does its own internal cross-validation (randomly splitting the training data into
folds), which would reintroduce exactly the random-split temporal leakage `temporal_split` exists to
avoid (see [leakage](#leakage-temporal--data-leakage)). This only rules out `GridSearchCV`'s *default*
cross-validation, though — see [PredefinedSplit](#predefinedsplit) for how the project later reused
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

### MODIS
One of two satellite instrument families FIRMS distributes detections from (the other is
[VIIRS](#viirs)) — coarser resolution (~1km), longer historical record (~2000-present). See
[Data sources](02-data-sources.md#firms--fire-detections-the-label-source).

### ModelBundle
`training/persist.py::ModelBundle` — a small dataclass that bundles a fitted model together with the
exact `feature_columns` list (in order) it was trained on, plus free-form `metadata` (model type,
validation scores, when it was trained), `joblib`-serialized as one object. Exists so serving code
can never independently hardcode or guess the feature list and have it silently drift from what the
model actually expects. See [Serving](07-serving.md#persisting-a-model-without-losing-its-contract).

### Nearest-neighbor join
Matching each point in dataset A to whichever point in dataset B is spatially closest, rather than
requiring an exact coordinate match. Used to attach ERA5-Land's coarser weather grid to the finer
fire-detection grid. See [Weather join](04-weather-join.md#problem-1-two-different-grids).

### Panel data
Data with many distinct units (here, grid cells), each observed repeatedly over time — as opposed to
a single time series (one unit over time) or plain cross-sectional data (many units at one point in
time). Feature engineering here (lags, rolling windows) must be grouped by unit (`cell_id`) to stay
valid. See
[Feature engineering](05-feature-engineering.md#this-is-panel-data-not-a-single-time-series).

### PR-AUC (average precision)
Area under the precision-recall curve. Precision = of everything predicted positive, what fraction
really was; recall = of everything truly positive, what fraction got predicted. PR-AUC summarizes
the tradeoff across all thresholds into one number, and — unlike ROC-AUC — stays sensitive to model
quality even when negatives vastly outnumber positives, which is why it's the primary metric here.
See [Problem framing](01-problem-framing.md#the-metrics-used-instead).

### PredefinedSplit
An `sklearn.model_selection` cross-validation splitter that takes a fixed fold assignment instead of
computing one — you hand it a `test_fold` array (`-1` for rows that should always stay in training,
a fold index for rows that should be held out) and it reproduces exactly that split, no shuffling.
Used to let `RandomizedSearchCV`/`GridSearchCV` search hyperparameters against the project's existing
train/val split without their default cross-validation reshuffling the data and reintroducing the
same [leakage](#leakage-temporal--data-leakage) `temporal_split` exists to prevent. See
[Modeling & evaluation](06-modeling-and-evaluation.md#widening-the-search-randomizedsearchcv--predefinedsplit).

### Reanalysis
A dataset produced by running a physics-based weather/climate model *backward* over historical time,
constrained by whatever real observations exist, to produce a physically consistent, gap-free
estimate of conditions everywhere and every timestep — not raw observations, and not a forecast.
ERA5-Land is ECMWF's reanalysis product. See
[Data sources](02-data-sources.md#era5-land--weather-the-feature-source).

### ROC-AUC
Area under the ROC curve (true positive rate vs false positive rate across thresholds). Reported as
secondary context in this project — its false-positive-rate axis gets diluted by the huge number of
true negatives under class imbalance, so it can look deceptively good even when precision is poor.
See [Problem framing](01-problem-framing.md#the-metrics-used-instead).

### scale_pos_weight
`XGBClassifier`'s way of handling [class imbalance](#class-imbalance) — `XGBoost` has no
`class_weight="balanced"` option like `LogisticRegression`/`RandomForestClassifier`, so
`training/advanced_models.py::fit_xgboost` computes the equivalent manually:
`scale_pos_weight = (count of negatives) / (count of positives)`, **recomputed from the training
fold only** each time a model is fit (never hardcoded, and never computed from validation/test), so
the correction can't leak information about val/test's class balance into training. See
[Modeling & evaluation](06-modeling-and-evaluation.md#randomforest-and-xgboost-trainingadvanced_modelspy).

### Top-k% capture
Of all actual positives, the fraction that fall within the model's top-k%-scored predictions. Framed
operationally: if only the top 10% riskiest cell-days could be acted on (limited crews/resources),
what fraction of real fires would that have caught? See
[Problem framing](01-problem-framing.md#the-metrics-used-instead).

### VIIRS
The satellite instrument family (on Suomi NPP / NOAA-20) this project's FIRMS source
(`VIIRS_SNPP_SP`) uses — higher spatial resolution (~375m) than MODIS, detections since ~2012. See
[Data sources](02-data-sources.md#firms--fire-detections-the-label-source).
