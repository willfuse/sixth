import pytest

from sixth.data import Bars, load_csv, synthetic_bars, synthetic_true_regimes


def test_synthetic_is_deterministic():
    a = synthetic_bars(n=300, seed=5)
    b = synthetic_bars(n=300, seed=5)
    assert a.close == b.close
    assert a.fingerprint() == b.fingerprint()


def test_different_seeds_differ():
    assert synthetic_bars(n=300, seed=1).close != synthetic_bars(n=300, seed=2).close


def test_ohlc_invariants_hold():
    b = synthetic_bars(n=500)
    for i in range(len(b)):
        assert b.high[i] >= max(b.open[i], b.close[i]) - 1e-9
        assert b.low[i] <= min(b.open[i], b.close[i]) + 1e-9
        assert b.close[i] > 0


def test_dates_are_business_days_and_ascending():
    import datetime
    b = synthetic_bars(n=200)
    dates = [datetime.date.fromisoformat(d) for d in b.dates]
    assert dates == sorted(dates)
    assert all(d.weekday() < 5 for d in dates)


def test_returns_start_at_zero_and_match_closes():
    b = synthetic_bars(n=50)
    r = b.returns()
    assert r[0] == 0.0
    assert r[5] == pytest.approx((b.close[5] - b.close[4]) / b.close[4])


def test_slice_preserves_alignment():
    b = synthetic_bars(n=200)
    s = b.slice(50, 100)
    assert len(s) == 50
    assert s.close == b.close[50:100]
    assert s.dates == b.dates[50:100]


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError):
        Bars("X", ["d1", "d2"], [1.0], [1.0], [1.0], [1.0, 2.0])


def test_volume_defaults_to_zeros():
    b = Bars("X", ["d1"], [1.0], [1.0], [1.0], [1.0])
    assert b.volume == [0.0]


def test_true_regimes_match_the_generated_length():
    from sixth.data import REGIME_SPECS
    labels = synthetic_true_regimes(n=400, seed=3)
    assert len(labels) == 400
    assert set(labels) <= set(REGIME_SPECS)


def test_true_regimes_switch_over_a_long_sample():
    """Regimes persist for 60-400 bars, so a short sample can legitimately be a
    single regime; a multi-year one cannot."""
    labels = synthetic_true_regimes(n=2520, seed=3)
    assert len(set(labels)) > 1


def test_regime_runs_are_persistent_not_iid():
    labels = synthetic_true_regimes(n=2520, seed=3)
    switches = sum(1 for a, b in zip(labels, labels[1:]) if a != b)
    assert switches < len(labels) / 50, "regimes must persist, not flip every bar"


def test_load_csv_roundtrip(tmp_path):
    p = tmp_path / "bars.csv"
    p.write_text("date,open,high,low,close,volume\n"
                 "2024-01-02,100,102,99,101,1000\n"
                 "2024-01-03,101,103,100,102,1100\n")
    b = load_csv(str(p), symbol="TEST")
    assert b.symbol == "TEST"
    assert b.close == [101.0, 102.0]
    assert b.volume == [1000.0, 1100.0]


def test_load_csv_falls_back_to_close_for_missing_ohlc(tmp_path):
    p = tmp_path / "close_only.csv"
    p.write_text("date,close\n2024-01-02,101\n2024-01-03,102\n")
    b = load_csv(str(p))
    assert b.open == b.close == [101.0, 102.0]


def test_load_csv_requires_a_close_column(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("date,price\n2024-01-02,101\n")
    with pytest.raises(ValueError, match="close"):
        load_csv(str(p))


def test_fingerprint_changes_with_data():
    a = synthetic_bars(n=100, seed=1)
    b = Bars(a.symbol, a.dates, a.open, a.high, a.low,
             [c * 1.01 for c in a.close], a.volume)
    assert a.fingerprint() != b.fingerprint()
