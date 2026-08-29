"""Sequence-modeling experiment: raw daily weather sequence vs. rolling-window features.

`research/neural-networks.md` argues against a neural network replacing the served RandomForest,
but names one legitimately open, falsifiable question it doesn't rule out: does a model that sees
the *raw* last-30-days weather sequence per cell (instead of hand-engineered rolling summaries like
`t2m_mean_7d`/`precip_30d`) capture a nonlinear temporal *shape* those summaries flatten away? This
module is that experiment, scoped exactly as the research doc recommended: a real comparison against
the current RandomForest, refit on the *same* rows under the *same* temporal-split discipline, not a
model adopted on principle.

This is a diagnostic/comparison script, matching the role `advanced_models.py` already plays for
RF/XGBoost tuning, it does not touch `export_model.py` or serving. Promoting a result (either
direction) into the served model is a separate, later decision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from firesight.evaluation.metrics import pr_auc, roc_auc, top_k_capture
from firesight.training.baseline import DATE_COLUMN, LABEL_COLUMN

RANDOM_STATE = 0

# The 5 raw daily *weather* quantities in `baseline.py::FEATURE_COLUMNS`, excluding the columns
# already dropped there as dead weight (d2m, u10, v10, wind_dir_sin/cos) and excluding the
# *rolling* columns (precip_7d/30d, t2m_mean_7d, rh_mean_7d, days_since_rain) this experiment
# exists to test a raw-sequence replacement for.
#
# FEATURE_COLUMNS has since grown to 32, and the rest is deliberately not represented here:
# neighbor_fire_count_{1,3,7}d and the 19 fuel_type_* one-hots aren't daily weather series a CNN
# could read a temporal shape out of. So the RandomForest comparison in __main__ is not a
# like-for-like *input* set (RF sees all 32 columns, the CNN sees these 5 raw series) -- it's
# matched on rows, not features, because the question is whether the raw sequence carries temporal
# signal the rolling summaries flatten away, not which model is better overall.
CHANNELS = ["t2m", "precip_mm", "swvl1", "relative_humidity", "wind_speed"]

# Matches the longest existing rolling window (precip_30d) and the question as posed in
# research/neural-networks.md.
SEQ_LEN = 30


def build_raw_sequences(
    df: pd.DataFrame,
    channels: list[str] = CHANNELS,
    seq_len: int = SEQ_LEN,
    cell_col: str = "cell_id",
    date_col: str = DATE_COLUMN,
) -> tuple[np.ndarray, pd.DataFrame]:
    """For every row with `seq_len` immediately-preceding, date-contiguous same-cell rows, build
    the raw `channels` sequence ending at (and including) that row.

    A row without a full, gap-free `seq_len`-day same-cell history is silently excluded rather than
    raised on or interpolated, the same "don't guess, just drop it" precedent
    `features/engineering.py::drop_incomplete_history` already established for insufficient rolling
    history. Built per cell (not as one global sliding window) so a window can never mix two cells'
    days together.

    Returns `(sequences, valid_rows)`: a `(N, seq_len, len(channels))` float32 array and the
    `N`-row slice of `df` each sequence ends on, in the same order, `valid_rows` keeps its
    original positional index into `df` (not reset), so callers can align further row-level
    filtering/splitting (see this module's `__main__`) back to `sequences` by `.index`.
    """
    df = df.sort_values([cell_col, date_col])

    sequence_chunks = []
    row_index_chunks = []
    for _, group in df.groupby(cell_col, sort=True):
        if len(group) < seq_len:
            continue
        values = group[channels].to_numpy(dtype=np.float32)
        dates = group[date_col].to_numpy()

        windows = np.lib.stride_tricks.sliding_window_view(values, seq_len, axis=0)
        windows = np.moveaxis(windows, -1, 1)  # (n - seq_len + 1, channels, seq_len) -> (..., seq_len, channels)

        day_gaps = np.diff(dates).astype("timedelta64[D]").astype(int)
        gap_windows = np.lib.stride_tricks.sliding_window_view(day_gaps, seq_len - 1)
        contiguous = (gap_windows == 1).all(axis=1)

        end_positions = np.arange(seq_len - 1, len(group))[contiguous]
        sequence_chunks.append(windows[contiguous])
        row_index_chunks.append(group.index.to_numpy()[end_positions])

    if not sequence_chunks:
        return np.empty((0, seq_len, len(channels)), dtype=np.float32), df.iloc[0:0]

    sequences = np.concatenate(sequence_chunks, axis=0)
    valid_index = np.concatenate(row_index_chunks)
    valid_rows = df.loc[valid_index]
    return sequences, valid_rows


class SequenceCNN(nn.Module):
    """Small 1D-CNN over the time axis: two conv layers, global average pool, a small dense head.

    Not an LSTM/attention model, cheapest to train and matches the project's "start simple" bias;
    the temporal patterns worth testing here (a heat ramp, a compound dry-then-hot stretch) are
    local-window patterns a CNN's receptive field covers well, not long-range dependencies that
    would justify recurrence/attention's added complexity and slower training.
    """

    def __init__(self, n_channels: int, hidden_channels: int = 16):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, hidden_channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, channels) -> (batch, channels, seq_len), what Conv1d expects.
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x).squeeze(-1)  # raw logit, shape (batch,)


class AttentionPoolSequenceCNN(nn.Module):
    """`SequenceCNN` with its pooling layer swapped for a learned, per-day softmax-weighted sum.

    Narrower than a full self-attention/Transformer block (see
    docs/06-modeling-and-evaluation.md#4-attention-pooling-on-the-sequence-model-a-narrower-angle-than-a-full-transformer
    for why that was rejected: this project has twice shown more model capacity backfires here).

    `conv1`/`conv2` are byte-for-byte unchanged from `SequenceCNN`, only `AdaptiveAvgPool1d(1)` is
    replaced with one `nn.Linear(hidden_channels*2, 1)` scoring each day + a softmax over the time
    axis, 33 extra parameters (3,553 -> 3,586 at the default `hidden_channels=16`, since pooling
    itself has none). That isolates a genuinely different, still-open question from the one
    `SequenceCNN` already answered: does *learning which days to weight* beat uniform averaging,
    independent of the raw-sequence-vs-rolling-features comparison.
    """

    def __init__(self, n_channels: int, hidden_channels: int = 16):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, hidden_channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=5, padding=2)
        self.attention_score = nn.Linear(hidden_channels * 2, 1)
        self.fc1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_channels, 1)
        # Populated by the most recent forward() call -- (batch, seq_len), softmax-normalized per
        # row. Exists purely for the interpretability artifact (attention-weight-by-lag-day below);
        # scoring/training never reads it back.
        self.last_attention_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, channels) -> (batch, channels, seq_len), what Conv1d expects.
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))  # (batch, hidden*2, seq_len)
        x = x.transpose(1, 2)  # (batch, seq_len, hidden*2) -- one feature vector per day

        scores = self.attention_score(x).squeeze(-1)  # (batch, seq_len)
        weights = torch.softmax(scores, dim=1)
        self.last_attention_weights = weights.detach()

        pooled = (x * weights.unsqueeze(-1)).sum(dim=1)  # (batch, hidden*2)
        h = torch.relu(self.fc1(pooled))
        h = self.dropout(h)
        return self.fc2(h).squeeze(-1)  # raw logit, shape (batch,)


def fit_sequence_cnn(
    train_seq: np.ndarray,
    train_y: np.ndarray,
    val_seq: np.ndarray,
    val_y: np.ndarray,
    epochs: int = 8,
    batch_size: int = 4096,
    lr: float = 1e-3,
    random_state: int = RANDOM_STATE,
    model_cls: type[nn.Module] = SequenceCNN,
) -> nn.Module:
    """Train `model_cls` (default `SequenceCNN`), tracking val PR-AUC each epoch and keeping the
    best-val state dict.

    `model_cls` is parameterized (not hardcoded to `SequenceCNN`) so `AttentionPoolSequenceCNN`
    below can reuse this exact training loop, the docs/06 proposal #4 point of that experiment is
    to isolate "does the pooling layer matter," so everything else (loss, optimizer, epochs, model
    selection) must be held identical, not reimplemented in parallel.

    `pos_weight` in `BCEWithLogitsLoss` is PyTorch's version of `class_weight="balanced"` elsewhere
    in this project: it's the train fold's own negative/positive ratio, so the loss doesn't just
    learn to always predict "no fire." Model *selection* (which epoch's weights to keep) uses val,
    never test, the same discipline `tune_model`/`tune_random_search` already enforce for the
    tree-based models.
    """
    torch.manual_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_cls(n_channels=train_seq.shape[-1]).to(device)
    negative = int((train_y == 0).sum())
    positive = int((train_y == 1).sum())
    pos_weight = torch.tensor(negative / max(positive, 1), dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_seq_t = torch.from_numpy(train_seq)
    train_y_t = torch.from_numpy(train_y.astype(np.float32))
    val_seq_t = torch.from_numpy(val_seq).to(device)

    n = len(train_seq_t)
    best_val_pr_auc = -float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb = train_seq_t[idx].to(device)
            yb = train_y_t[idx].to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        train_loss = total_loss / n

        model.eval()
        with torch.no_grad():
            val_scores = torch.sigmoid(model(val_seq_t)).cpu().numpy()
        val_pr_auc_score = pr_auc(val_y, val_scores)
        print(f"epoch {epoch}/{epochs}: train_loss={train_loss:.5f} val_pr_auc={val_pr_auc_score:.5f}", flush=True)

        if val_pr_auc_score > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.to(device)


def score_sequence_model(model: SequenceCNN, sequences: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    """Same `{pr_auc, roc_auc, top_10pct_capture}` shape `baseline.py::score_model` returns."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        y_score = torch.sigmoid(model(torch.from_numpy(sequences).to(device))).cpu().numpy()
    return {
        "pr_auc": pr_auc(y_true, y_score),
        "roc_auc": roc_auc(y_true, y_score),
        "top_10pct_capture": top_k_capture(y_true, y_score, k_fraction=0.1),
    }


if __name__ == "__main__":
    from firesight.training.advanced_models import fit_random_forest
    from firesight.training.baseline import (
        DATASET_PATH,
        FIRE_SEASON_END,
        TRAIN_END,
        VAL_END,
        filter_fire_season,
        score_model,
        temporal_split,
    )
    from firesight.training.export_model import BEST_RANDOM_FOREST_PARAMS

    # 30-day margin before fire season start so every fire-season target date's lookback window is
    # available, without building (and holding in memory) sequences for irrelevant winter/shoulder
    # months no target row will ever come from.
    SEQUENCE_LOOKBACK_START_MD = "04-01"

    df = pd.read_parquet(DATASET_PATH)
    month_day = df[DATE_COLUMN].dt.strftime("%m-%d")
    lookback_window = df[(month_day >= SEQUENCE_LOOKBACK_START_MD) & (month_day <= FIRE_SEASON_END)]

    print("building raw sequences...", flush=True)
    sequences, valid_rows = build_raw_sequences(lookback_window)
    valid_rows = valid_rows.reset_index(drop=True)
    sequence_by_position = sequences  # aligned 1:1 with valid_rows' new positional index

    season_rows = filter_fire_season(valid_rows)  # index preserved -> still aligns to sequence_by_position
    train_rows, val_rows, test_rows = temporal_split(season_rows, TRAIN_END, VAL_END)

    train_seq = sequence_by_position[train_rows.index.to_numpy()]
    val_seq = sequence_by_position[val_rows.index.to_numpy()]
    test_seq = sequence_by_position[test_rows.index.to_numpy()]
    train_y = train_rows[LABEL_COLUMN].to_numpy()
    val_y = val_rows[LABEL_COLUMN].to_numpy()
    test_y = test_rows[LABEL_COLUMN].to_numpy()

    print(f"train={len(train_rows):,} (positives={int(train_y.sum())})", flush=True)
    print(f"val={len(val_rows):,} (positives={int(val_y.sum())})", flush=True)
    print(f"test={len(test_rows):,} (positives={int(test_y.sum())})", flush=True)

    print("\n--- SequenceCNN ---", flush=True)
    cnn = fit_sequence_cnn(train_seq, train_y, val_seq, val_y)
    print("CNN val (2023): ", score_sequence_model(cnn, val_seq, val_y), flush=True)
    print("CNN test (2024, untouched during training):", score_sequence_model(cnn, test_seq, test_y), flush=True)

    # Refit RandomForest on the exact same row subset (not the docs/06 numbers, which are computed
    # on a slightly different row set) so the comparison is apples-to-apples on identical rows.
    # Uses the same tuned BEST_RANDOM_FOREST_PARAMS the served model uses, not sklearn's untuned
    # defaults (unbounded depth), research/neural-networks.md already documented that unbounded
    # depth on this dataset overfits and craters top-10% capture, which would make this comparison
    # meaningless.
    print("\n--- RandomForest (same rows, tuned params) ---", flush=True)
    rf = fit_random_forest(train_rows, **BEST_RANDOM_FOREST_PARAMS)
    print("RF val (2023): ", score_model(rf, val_rows), flush=True)
    print("RF test (2024, untouched during training):", score_model(rf, test_rows), flush=True)

    # Attention-pooling experiment (docs/06 proposal #4): cheap, low-risk, pitched as research
    # completeness/interpretability, not a likely path to beating RandomForest -- this project is
    # already 2-for-2 against "more model capacity helps" on this data (SequenceCNN above, and an
    # earlier uncapped-depth RandomForest diagnostic). Reuses the exact same fit_sequence_cnn/
    # score_sequence_model harness as SequenceCNN, isolating the pooling layer as the one variable.
    print("\n--- AttentionPoolSequenceCNN ---", flush=True)
    attn_cnn = fit_sequence_cnn(train_seq, train_y, val_seq, val_y, model_cls=AttentionPoolSequenceCNN)
    print("Attention-pool val (2023): ", score_sequence_model(attn_cnn, val_seq, val_y), flush=True)
    print(
        "Attention-pool test (2024, untouched during training):",
        score_sequence_model(attn_cnn, test_seq, test_y),
        flush=True,
    )

    # Interpretability artifact: does the learned weighting concentrate near the ignition day, or
    # spread out evenly (== learned to reproduce plain averaging, a negative result per docs/06's
    # own framing)? Mean attention by lag-day across every real test-set positive, not just a
    # hand-picked few, so the table isn't cherry-picked toward whichever examples look interesting.
    device = next(attn_cnn.parameters()).device
    attn_cnn.eval()
    with torch.no_grad():
        test_positive_seq = test_seq[test_y == 1]
        _ = attn_cnn(torch.from_numpy(test_positive_seq).to(device))
        mean_attention_by_lag = attn_cnn.last_attention_weights.mean(dim=0).cpu().numpy()

    print(f"\n=== mean attention weight by lag-day, {len(test_positive_seq)} real test-set fires ===", flush=True)
    uniform_weight = 1.0 / SEQ_LEN
    lag_table = pd.DataFrame(
        {
            "days_before_target": np.arange(SEQ_LEN - 1, -1, -1),
            "mean_attention": mean_attention_by_lag,
        }
    )
    print(lag_table.to_string(index=False), flush=True)
    ignition_day_weight = float(mean_attention_by_lag[-1])
    print(f"\nuniform (plain-averaging) weight per day would be: {uniform_weight:.4f}", flush=True)
    print(f"learned weight on the ignition day itself (lag 0): {ignition_day_weight:.4f}", flush=True)
    print(
        "-> "
        + (
            "concentrates near the ignition day: real, learned weighting."
            if ignition_day_weight > uniform_weight * 1.5
            else "close to uniform: learned to reproduce plain averaging, a negative result per docs/06 #4."
        ),
        flush=True,
    )

    # Per-fire plot: attention weight vs. day-in-window, for a mix of caught and missed test-set
    # fires (using this model's own top-10%-by-score cutoff, matching top_k_capture's definition)
    # -- shows whether the *shape* of the learned weighting differs between fires the model ranks
    # highly and ones it doesn't, not just the pooled average above.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with torch.no_grad():
        test_scores = torch.sigmoid(attn_cnn(torch.from_numpy(test_seq).to(device))).cpu().numpy()
    top_10pct_cutoff = np.quantile(test_scores, 0.9)
    positive_idx = np.flatnonzero(test_y == 1)
    caught_idx = positive_idx[test_scores[positive_idx] >= top_10pct_cutoff]
    missed_idx = positive_idx[test_scores[positive_idx] < top_10pct_cutoff]
    rng = np.random.default_rng(0)
    example_idx = np.concatenate(
        [
            rng.choice(caught_idx, size=min(5, len(caught_idx)), replace=False) if len(caught_idx) else [],
            rng.choice(missed_idx, size=min(5, len(missed_idx)), replace=False) if len(missed_idx) else [],
        ]
    ).astype(int)

    with torch.no_grad():
        _ = attn_cnn(torch.from_numpy(test_seq[example_idx]).to(device))
        example_weights = attn_cnn.last_attention_weights.cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    lag_days = np.arange(SEQ_LEN - 1, -1, -1)
    for row_pos, weights in zip(example_idx, example_weights):
        caught = test_scores[row_pos] >= top_10pct_cutoff
        ax.plot(lag_days, weights, marker="o", markersize=3, alpha=0.7, color="#1d4ed8" if caught else "#b91c1c")
    ax.plot([], [], color="#1d4ed8", label="caught (top 10%)")
    ax.plot([], [], color="#b91c1c", label="missed")
    ax.invert_xaxis()
    ax.set_xlabel("days before target (0 = ignition day)")
    ax.set_ylabel("attention weight")
    ax.set_title("AttentionPoolSequenceCNN: per-fire attention weight by lag-day")
    ax.legend()
    out_path = DATASET_PATH.parent / "attention_weights_by_fire.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\nsaved {out_path}", flush=True)
