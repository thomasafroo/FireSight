# Problem framing & metrics

## What we're actually predicting

Not "will BC have a wildfire this year" — that's certain, and useless. The prediction target is much
narrower and defined per **(grid cell, day)**:

> Given conditions up to and including day *D*, in grid cell *C*, will a real vegetation fire be
> detected in that cell on day *D* (or, once the model matures, within the next *N* days)?

**Update 2026-08-21: the multi-day-ahead extension was tried.** `ignited_next_3d`
([Grid & labels](03-grid-and-labels.md#the-multi-day-ahead-label-ignited_next_nd-2026-08-21)) is a real,
working label with skill well above the dummy floor, but a genuine accuracy gap versus same-day
prediction — see [Modeling &
evaluation](06-modeling-and-evaluation.md#testing-the-multi-day-ahead-label-2026-08-21) for the full
result. Validated, not yet served.

That's a **binary classification** problem, not regression: the label (`ignited`) is 0 or 1, not a
continuous quantity. We're not predicting *how big* a fire will be or *how much* area burns — just
whether ignition is detected at all. Fire size/spread modeling is a different (harder) problem,
deliberately out of scope for the MVP.

Why split the region into a grid instead of predicting for "Kamloops" as a whole? Because risk isn't
uniform across a region — a river valley and a dry ridge 20km apart can have very different risk on
the same day. Gridding is how continuous space gets turned into discrete rows a classifier can be
trained on. See [Grid & labels](03-grid-and-labels.md) for how the grid itself is built.

## Why this is a *rare-event* classification problem

Real vegetation fire detections in the Kamloops bbox, filtered and gridded (see
`pipeline/build_dataset.py`), come out to about **0.16% of (cell, day) rows** having `ignited == 1`.
Fires are, thankfully, rare. This has a direct consequence for how the model should be built and
evaluated:

**Accuracy is a meaningless — actually actively misleading — metric here.** A model that outputs "no
fire" for every single row scores 99.84% accuracy while catching zero fires. It would look almost
perfect on the one metric most people reach for first, while being completely useless for the actual
goal (warning about fires before they're detected some other way). This is why
`evaluation/metrics.py` doesn't even expose an accuracy function.

## The metrics used instead

All three are in `evaluation/metrics.py`:

- **PR-AUC** (`pr_auc`, area under the precision-recall curve, aka *average precision*) — the
primary metric. It only cares about how well the model ranks and scores the *positive* class, which
is exactly what matters when positives are rare: precision (of everything flagged as risky, how much
really burned) and recall (of everything that burned, how much was flagged) both stay meaningful
even at 0.16% base rate. See [glossary.md](glossary.md#pr-auc) for the full definition.
- **ROC-AUC** (`roc_auc`) — reported alongside PR-AUC for context, but trusted less. ROC-AUC's
false-positive-rate axis is diluted by the huge number of true negatives in an imbalanced problem,
so it can look good even when precision is poor. Kept as a secondary check, not the metric a
modeling decision gets made on.
- **Top-k% capture** (`top_k_capture`) — "if you could only act on the riskiest 10% of cell-days,
what fraction of actual fires would you have caught?" This is the metric closest to how the model
would actually be used operationally: fire management has finite resources (crews, aircraft,
patrols), so what matters isn't a probability threshold in the abstract — it's whether the
highest-ranked predictions are the ones that matter. Defaults to `k_fraction=0.1` (top 10%).

## Methodology: baseline first, complexity only if earned

Documented in the README's design notes, worth restating here because it's the single biggest
guardrail against wasted effort: every model gets compared against the *previous, simpler* model
before being kept.

`DummyClassifier` → `LogisticRegression` (class-weighted) → `RandomForestClassifier` → `XGBoost` →
(maybe, eventually) a neural net.

The `DummyClassifier` isn't a throwaway step — it's the floor. If logistic regression doesn't
clearly beat a classifier that ignores the features entirely, something upstream (features, labels,
join) is broken, and no amount of model complexity will fix that. Complexity is justified by a
measured PR-AUC/top-k gain, not by "trees are usually better than linear models."
