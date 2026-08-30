import pytest

from sixth.backtest import Costs
from sixth.breaker import break_it
from sixth.data import Bars, synthetic_bars
from sixth.strategy import Strategy, get


def test_flat_strategy_is_indistinguishable_from_nothing(bars):
    r = break_it(get("flat"), bars, null_samples=20)
    assert r.baseline_sharpe == 0.0
    assert r.cost_death_bps is None


def test_cost_death_point_is_found_for_a_turnover_strategy(bars):
    r = break_it(get("mean_reversion"), bars, null_samples=10)
    if r.baseline_sharpe > 0:
        assert r.cost_death_bps is not None and r.cost_death_bps > 0


def test_buy_hold_survives_cost_scaling(bars):
    """No turnover means no cost sensitivity."""
    r = break_it(get("buy_hold"), bars, null_samples=10)
    assert all(p.survived for p in r.probes if p.name.startswith("costs_x"))


def test_buy_hold_matches_the_asset_in_its_worst_windows(bars):
    r = break_it(get("buy_hold"), bars, null_samples=10)
    windows = [p for p in r.probes if p.name.startswith("worst_window")]
    assert windows
    assert all(p.survived for p in windows), "holding cannot underperform holding"


def test_fatal_regimes_are_collected(bars):
    r = break_it(get("mean_reversion"), bars, null_samples=10)
    assert isinstance(r.fatal_regimes, list)
    for label in r.fatal_regimes:
        probe = next(p for p in r.probes if p.name == f"regime_{label}")
        assert not probe.survived


def test_null_p_value_is_a_probability(bars):
    r = break_it(get("sma_cross"), bars, null_samples=50)
    assert 0.0 < r.null_p_value <= 1.0


def test_random_signal_does_not_beat_the_random_null(bars):
    """A coin flip must not be certified as an edge."""
    r = break_it(get("random_signal", seed=3), bars, null_samples=100)
    probe = next(p for p in r.probes if p.name == "vs_random_null")
    assert not probe.survived


def test_lag_probe_kills_a_strategy_that_peeks(bars):
    """A strategy whose weight is next bar's direction is pure lookahead; one
    extra bar of delay must destroy it."""
    closes = bars.close
    peek = Strategy("peek", lambda b: [1.0 if i + 1 < len(b) and b.close[i + 1] > b.close[i]
                                       else 0.0 for i in range(len(b))])
    r = break_it(peek, bars, Costs(0, 0, 0, 0), null_samples=10)
    lag = next(p for p in r.probes if p.name == "execution_lag_1bar")
    assert r.baseline_sharpe > 1.0, "the peeking strategy should look spectacular"
    assert lag.metric < r.baseline_sharpe / 2, "delay must destroy a lookahead edge"


def test_robustness_is_a_fraction(bars):
    r = break_it(get("sma_cross"), bars, null_samples=10)
    assert 0.0 <= r.robustness <= 1.0
    assert len(r.failures) == sum(1 for p in r.probes if not p.survived)


def test_report_serialises(bars):
    import json
    r = break_it(get("sma_cross"), bars, null_samples=10)
    assert json.loads(json.dumps(r.to_dict()))["strategy"] == "sma_cross"


def test_param_probe_is_skipped_when_there_are_no_numeric_params(bars):
    r = break_it(get("buy_hold"), bars, null_samples=10)
    assert not any(p.name == "param_neighbourhood" for p in r.probes)
