"""Sequence-modeling experiment: raw daily weather sequence vs. rolling-window features.

`research/neural-networks.md` argues against a neural network replacing the served RandomForest,
but names one legitimately open, falsifiable question it doesn't rule out: does a model that sees
the *raw* last-30-days weather sequence per cell (instead of hand-engineered rolling summaries like
`t2m_mean_7d`/`precip_30d`) capture a nonlinear temporal *shape* those summaries flatten away? This
module is that experiment, scoped exactly as the research doc recommended: a real comparison against
the current RandomForest, refit on the *same* rows under the *same* temporal-split discipline, not a
model adopted on principle.

This is a diagnostic/comparison script, matching the role `advanced_models.py` already plays for
RF/XGBoost tuning — it does not touch `export_model.py` or serving. Promoting a result (either
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

# The 5 raw daily quantities already in `baseline.py::FEATURE_COLUMNS`, excluding the columns
# already dropped there as dead weight (d2m, u10, v10, wind_dir_sin/cos) and excluding the
# *rolling* columns (precip_7d/30d, t2m_mean_7d, rh_mean_7d, days_since_rain) this experiment
# exists to test a raw-sequence replacement for.
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
    raised on or interpolated — the same "don't guess, just drop it" precedent
    `features/engineering.py::drop_incomplete_history` already established for insufficient rolling
    history. Built per cell (not as one global sliding window) so a window can never mix two cells'
    days together.

    Returns `(sequences, valid_rows)`: a `(N, seq_len, len(channels))` float32 array and the
    `N`-row slice of `df` each sequence ends on, in the same order — `valid_rows` keeps its
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

    Not an LSTM/attention model — cheapest to train and matches the project's "start simple" bias;
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


def fit_sequence_cnn(
    train_seq: np.ndarray,
    train_y: np.ndarray,
    val_seq: np.ndarray,
    val_y: np.ndarray,
    epochs: int = 8,
    batch_size: int = 4096,
    lr: float = 1e-3,
    random_state: int = RANDOM_STATE,
) -> SequenceCNN:
    """Train `SequenceCNN`, tracking val PR-AUC each epoch and keeping the best-val state dict.

    `pos_weight` in `BCEWithLogitsLoss` is PyTorch's version of `class_weight="balanced"` elsewhere
    in this project: it's the train fold's own negative/positive ratio, so the loss doesn't just
    learn to always predict "no fire." Model *selection* (which epoch's weights to keep) uses val,
    never test — the same discipline `tune_model`/`tune_random_search` already enforce for the
    tree-based models.
    """
    torch.manual_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SequenceCNN(n_channels=train_seq.shape[-1]).to(device)
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
    # defaults (unbounded depth) — research/neural-networks.md already documented that unbounded
    # depth on this dataset overfits and craters top-10% capture, which would make this comparison
    # meaningless.
    print("\n--- RandomForest (same rows, tuned params) ---", flush=True)
    rf = fit_random_forest(train_rows, **BEST_RANDOM_FOREST_PARAMS)
    print("RF val (2023): ", score_model(rf, val_rows), flush=True)
    print("RF test (2024, untouched during training):", score_model(rf, test_rows), flush=True)
