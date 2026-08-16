# Neural networks for FireSight: a feasibility writeup

Research notes, not a decision — the README's original 10-step plan named "Neural Network" as the
step after XGBoost, and the project status page still lists it as open. This is due diligence on
whether that step is actually worth taking, given what's been learned since that plan was written:
RandomForest/XGBoost already clear the Dummy/LogisticRegression floors by a wide margin, permutation
importance shows the served model leans on only ~6 real features, and a capacity increase has already
been tried once (the uncapped-depth diagnostic in
[docs/06](../docs/06-modeling-and-evaluation.md#known-limitation-a-wintershoulder-season-blind-spot))
and made things worse, not better. None of this is committed anywhere else in the repo — it lives here
because it's a feasibility argument, not a pipeline stage `docs/` documents.

## The general case against, and why it applies unusually well here

The broader ML literature on tabular data is fairly settled: tree ensembles (RandomForest, XGBoost,
CatBoost, LightGBM) consistently beat deep learning on small-to-medium structured/tabular datasets,
and the gap doesn't close just from more recent architectures. Grinsztajn et al. 2022, ["Why do
tree-based models still outperform deep learning on tabular
data?"](https://arxiv.org/abs/2207.08815) is the reference result here: tree-based models remain
state-of-the-art on datasets around 10K rows even without accounting for their much faster training,
and the gap is attributed to tabular data's irregular, non-smooth target functions (trees split on
thresholds naturally; neural nets have to learn that same behavior from scratch via smooth
activations) and to many uninformative features diluting a dense network's shared representation. A
2025 comprehensive benchmark across 111 datasets found CatBoost winning 17.1% of the time with the
first deep-learning model not appearing until fifth place at 9.9% ([ScienceDirect,
2025](https://www.sciencedirect.com/science/article/pii/S0925231225020090)); more recent tabular
architectures like TabM ([Gorishniy et al.,
2024](https://arxiv.org/abs/2410.24210)) narrow the gap somewhat via parameter-efficient ensembling,
but the newest work in this space is still framed as *narrowing the gap to* tree ensembles, not
routinely beating well-tuned ones.

Three things about FireSight's specific situation make this general pattern apply even more strongly
than average, not less:

1. **Very few features (10, after [dropping dead
   weight](../docs/06-modeling-and-evaluation.md#dropping-the-dead-weight-features)), all numeric, no
   high-cardinality categoricals.** A neural network's main structural advantage over trees is
   automatic feature interaction and representation learning when there's a lot of raw signal to
   compress (images, text, many correlated or high-cardinality columns). With 10 plain numeric
   weather features, there's very little for a network to discover that a depth-6 RandomForest with
   238 trees can't already find by splitting — and permutation importance shows the model is already
   extracting essentially all the available signal from just soil moisture, 30-day precip, 7-day mean
   temp, raw temp, 7-day mean humidity, and wind speed.
2. **Extreme class imbalance on very few positive examples.** Train has on the order of a few
   thousand fire-days out of millions of rows (~0.16-0.2% positive rate). Neural networks are
   generally more data-hungry and more prone to unstable optimization/poor calibration at this scale
   of rare positives than tree ensembles with a class-imbalance correction — the project's own
   baseline-first results already show `class_weight="balanced"` RandomForest/XGBoost handling this
   robustly, which is exactly the comparison [Problem
   framing](../docs/01-problem-framing.md#methodology-baseline-first-complexity-only-if-earned)'s
   methodology exists to force before adding complexity.
3. **The known ceiling here is a data-generating-process problem, not a model-capacity problem.** The
   [winter/shoulder-season blind
   spot](../docs/06-modeling-and-evaluation.md#known-limitation-a-wintershoulder-season-blind-spot)
   comes from winter fires being more often human-caused than weather-driven — there is no signal in
   *any* weather feature for that, by construction. A neural network cannot manufacture information
   that was never in the input. Worse, the one time more capacity was actually tried (an uncapped-
   depth, 400-tree diagnostic RF, run specifically to test whether the tuned model's shallow depth was
   the bottleneck) made the December/February blind spot barely move while *cratering* every other
   metric (test top-10% capture 71.9% -> 26.4%) — the classic overfit-to-majority-class-noise failure
   mode. A neural network, with even less inductive bias toward the kind of threshold splits this
   problem rewards and even more raw parameter capacity, would be expected to hit the same wall at
   least as easily.

## Where a neural network could plausibly still help — and the honest case against each

Not every neural-network angle is ruled out by the above; a straight "swap RandomForest for an MLP
classifier" is, but a few more targeted ideas are worth naming and then honestly weighing:

- **Sequence modeling of raw daily weather** (an LSTM, 1D-CNN, or small attention model consuming the
  last N days of raw `t2m`/`precip_mm`/`swvl1`/... per cell, instead of hand-engineered rolling
  windows) could in principle capture a nonlinear temporal *shape* — e.g. a specific temperature ramp
  or a compound heat-then-dry pattern — that a fixed 7-day mean or 30-day sum flattens away. This is
  the most legitimate of the ideas here, because it changes what information the model sees rather
  than just how it's fit. But there's no positive evidence yet that the current rolling-window
  features are under-fitting the temporal signal — permutation importance shows `t2m_trend_7d` (the
  one existing feature that already tried to capture temporal *shape*, not just level) at essentially
  zero importance on both val and test, which is at least weak evidence against there being an
  easily-found temporal pattern the current features are missing, though it doesn't rule out a
  non-linear one a hand-written trend feature couldn't express.
- **Learned per-cell embeddings** instead of excluding `cell_id` entirely could in principle let
  nearby/similar cells share statistical strength without full one-hot memorization. But this doesn't
  address the actual reason `cell_id` was excluded (see [Where `ColumnTransformer`
  fits](../docs/06-modeling-and-evaluation.md#where-columntransformer-fits)): with only a few thousand
  positives spread across 1,443 cells, most cells have too few (often zero) recorded fires for any
  representation — embedded or not — to learn a reliable per-cell effect that generalizes to a future
  year, so this mainly trades one overfitting risk for a subtler version of the same one.
- **Tabular-specific deep architectures** (FT-Transformer, TabM, etc.) exist precisely to close the
  gap the literature above documents, and TabM specifically is a 2024 result aimed at getting
  ensemble-like robustness from a single network. But adopting one here means real added engineering
  cost — numeric-feature embeddings, more hyperparameters to tune safely under the project's
  temporal-split discipline (`PredefinedSplit`, never `GridSearchCV`'s default random CV — see
  [Widening the
  search](../docs/06-modeling-and-evaluation.md#widening-the-search-randomizedsearchcv--predefinedsplit)),
  and meaningfully slower iteration — for a class of model the current literature still frames as
  *narrowing the gap to* tuned GBMs on data this size and shape, not beating them outright.

## Recommendation

Don't add a neural network as a classifier to replace or ensemble with the served RandomForest right
now. Every piece of evidence already gathered in this project — the baseline-first comparison, the
feature-importance results, and the one direct capacity experiment that was run — points the same
direction: the bottleneck here is available information (weather-only features can't see
human-caused winter ignitions) and data volume (a few thousand positives), not model expressiveness,
and neural networks don't have an answer for either. This matches the general tabular-data literature
closely enough that it would be surprising for FireSight to be the exception.

If this gets revisited, the sequence-modeling-of-raw-weather idea above is the one angle that would
actually test a different hypothesis (temporal shape vs. temporal level) rather than just adding
capacity — but it should be scoped as a real experiment with a clear falsifiable question ("does a
model that sees the raw 30-day weather sequence beat one that sees `t2m_mean_7d`/`precip_30d` on the
*same* temporal train/val/test split, evaluated the same way"), benchmarked against the current
RandomForest under the exact same `PredefinedSplit`/test-set-never-touched-during-tuning discipline
the rest of this project already uses — not adopted on the general principle that "neural network" was
next on an old plan.
