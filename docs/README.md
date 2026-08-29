# FireSight ML Guide

This is the teaching companion to the code in `src/firesight/`. It exists because the *why* behind a
data pipeline disappears fast: six months from now, `// lat_size` in `grid.py` won't explain
itself. Each page pairs a concept with the exact function that implements it, so you can read the
reasoning and the code side by side.

**Who this is for:** written assuming you know `pandas`/`scikit-learn` basics (preprocessing,
`ColumnTransformer`, train/test splits) but are newer to geospatial data and time series-flavored
ML. Terms get defined the first time they're used; if something's undefined, check
[glossary.md](glossary.md).

## Pages

1. [Problem framing & metrics](01-problem-framing.md), what we're predicting, why it's a
classification problem, why accuracy is the wrong metric for it.
2. [Data sources](02-data-sources.md), FIRMS (fire detections) and ERA5-Land (weather reanalysis):
what they are, what their fields mean, why these two.
3. [Grid & labels](03-grid-and-labels.md), turning fire *points* into a *grid*, and turning
detections into a labeled (cell, date) table with real negatives, not just positives.
4. [Weather join](04-weather-join.md), matching a coarse weather grid to a finer fire grid, and the
accumulated-vs-instantaneous variable bug that would have silently corrupted the precipitation
feature.
5. [Feature engineering](05-feature-engineering.md), turning raw daily weather into the
lag/rolling/trend features a model can actually use.
6. [Modeling & evaluation](06-modeling-and-evaluation.md), temporal train/val/test splitting, the
baseline-first methodology, the metrics used instead of accuracy, and the measured results behind
every model and feature-set decision. Starts with [Current
model](06-modeling-and-evaluation.md#current-model), what is served today and what it scores; the
rest of the page is the dated history that got there.
7. [Serving](07-serving.md), persisting a model safely, the FastAPI inference/demo endpoints, and
the minimal frontend risk map.
8. [Future directions & open questions](08-future-directions.md), what's genuinely still open
(multi-day calibration, lightning data) versus not planned or out of scope (BC-wide expansion,
fire-spread modeling, auth), and why.
9. [Glossary](glossary.md), every term used across these pages, defined once.

## How this guide stays honest

Docs rot the moment they stop being updated alongside the code they describe. The rule for this
project (see the repo's `CLAUDE.md`): **any change to `src/firesight/` that changes what a module
does, why it does it that way, or a parameter/format it depends on, gets a matching doc update in
the same session.** If a page and the code disagree, the code is right and the page is stale, flag
it rather than trusting it.
