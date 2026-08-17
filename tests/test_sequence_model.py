import numpy as np
import pandas as pd

from firesight.training.sequence_model import (
    SequenceCNN,
    build_raw_sequences,
    fit_sequence_cnn,
    score_sequence_model,
)


def _make_cell_frame(cell_id: str, dates: list[str], x: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"cell_id": cell_id, "date": pd.to_datetime(dates), "x": x})


def test_build_raw_sequences_returns_windows_ending_on_each_row_with_full_history():
    cell_a = _make_cell_frame("A", ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"], [1, 2, 3, 4, 5])
    cell_b = _make_cell_frame("B", ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"], [10, 20, 30, 40, 50])
    df = pd.concat([cell_a, cell_b], ignore_index=True)

    sequences, valid_rows = build_raw_sequences(df, channels=["x"], seq_len=3)

    assert sequences.shape == (6, 3, 1)
    assert len(valid_rows) == 6

    row_a_day3 = valid_rows[(valid_rows["cell_id"] == "A") & (valid_rows["date"] == "2020-01-03")]
    assert len(row_a_day3) == 1
    seq_a_day3 = sequences[valid_rows.index.get_loc(row_a_day3.index[0])]
    assert seq_a_day3[:, 0].tolist() == [1.0, 2.0, 3.0]

    row_b_day5 = valid_rows[(valid_rows["cell_id"] == "B") & (valid_rows["date"] == "2020-01-05")]
    seq_b_day5 = sequences[valid_rows.index.get_loc(row_b_day5.index[0])]
    assert seq_b_day5[:, 0].tolist() == [30.0, 40.0, 50.0]


def test_build_raw_sequences_never_mixes_two_cells_history():
    cell_a = _make_cell_frame("A", ["2020-01-01", "2020-01-02", "2020-01-03"], [1, 2, 3])
    cell_b = _make_cell_frame("B", ["2020-01-01", "2020-01-02", "2020-01-03"], [100, 200, 300])
    df = pd.concat([cell_a, cell_b], ignore_index=True)

    sequences, valid_rows = build_raw_sequences(df, channels=["x"], seq_len=3)

    # Every window's values should belong entirely to one cell's own value range.
    for i, (_, row) in enumerate(valid_rows.iterrows()):
        window = sequences[i, :, 0]
        if row["cell_id"] == "A":
            assert window.max() <= 3
        else:
            assert window.min() >= 100


def test_build_raw_sequences_excludes_windows_spanning_a_date_gap():
    # day 4 is missing for cell C: 01, 02, 03, [gap], 05, 06, 07
    dates = ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-05", "2020-01-06", "2020-01-07"]
    df = _make_cell_frame("C", dates, [1, 2, 3, 4, 5, 6])

    sequences, valid_rows = build_raw_sequences(df, channels=["x"], seq_len=3)

    # Only two full, gap-free 3-day windows exist: (01,02,03)->x=[1,2,3] and (05,06,07)->x=[4,5,6].
    # The windows that would straddle the gap (ending on 05 or 06) must be excluded.
    assert len(valid_rows) == 2
    assert set(valid_rows["date"].dt.strftime("%Y-%m-%d")) == {"2020-01-03", "2020-01-07"}
    contents = sorted(sequences[:, :, 0].tolist())
    assert contents == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_build_raw_sequences_skips_cells_with_fewer_than_seq_len_rows():
    df = _make_cell_frame("A", ["2020-01-01", "2020-01-02"], [1, 2])
    sequences, valid_rows = build_raw_sequences(df, channels=["x"], seq_len=3)
    assert sequences.shape == (0, 3, 1)
    assert len(valid_rows) == 0


def test_fit_and_score_sequence_cnn_runs_end_to_end_on_synthetic_data():
    rng = np.random.default_rng(0)
    n_train, n_val, seq_len, n_channels = 300, 100, 10, 2
    train_seq = rng.normal(size=(n_train, seq_len, n_channels)).astype(np.float32)
    val_seq = rng.normal(size=(n_val, seq_len, n_channels)).astype(np.float32)
    train_y = (rng.uniform(size=n_train) < 0.1).astype(np.int64)
    val_y = (rng.uniform(size=n_val) < 0.1).astype(np.int64)

    model = fit_sequence_cnn(train_seq, train_y, val_seq, val_y, epochs=2, batch_size=64)
    assert isinstance(model, SequenceCNN)

    scores = score_sequence_model(model, val_seq, val_y)
    assert set(scores) == {"pr_auc", "roc_auc", "top_10pct_capture"}
    for value in scores.values():
        assert 0.0 <= value <= 1.0
