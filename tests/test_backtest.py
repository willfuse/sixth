import pytest

from sixth import backtest, strategy
from sixth.data import Bars, synthetic_bars


def flat_bars(n=50, price=100.0):
    return Bars("FLAT", [f"2020-01-{i%28+1:02d}" for i in range(n)],
                [price] * n, [price] * n, [price] * n, [price] * n)


def rising_bars(n=50, step=0.01):
    closes = [100.0 * (1 + step) ** i for i in range(n)]
    return Bars("UP", [f"d{i}" for i in range(n)], closes, closes, closes, closes)


def test_no_lookahead_signal_cannot_trade_its_own_bar():
    """A strategy that is long only on the single best bar must not capture it."""
    bars = rising_bars(20)
    spike = list(bars.close)
    spike[10] *= 1.5                       # one huge up bar
    bars = Bars("SPIKE", bars.dates, spike, spike, spike, spike)

    # Perfect hindsight: long exactly on the spike bar.
    s = strategy.Strategy("oracle", lambda b: [1.0 if i == 10 else 0.0
                                               for i in range(len(b))])
    res = backtest.run(s, bars, backtest.Costs(0, 0, 0, 0))
    # The weight set on bar 10 is held into bar 11, so the spike's own return is
    # not captured -- it is already in the past by the time the trade fills.
    assert res.returns[10] == 0.0


def test_flat_market_produces_zero_gross():
    res = backtest.run(strategy.get("buy_hold"), flat_bars(),
                       backtest.Costs(0, 0, 0, 0))
    assert sum(res.gross_returns) == pytest.approx(0.0, abs=1e-12)


def test_costs_reduce_returns_monotonically():
    bars = synthetic_bars(n=500)
    s = strategy.get("sma_cross")
    cheap = backtest.run(s, bars, backtest.Costs(0, 0, 0, 0))
    dear = backtest.run(s, bars, backtest.Costs(10, 10, 5, 0))
    assert sum(dear.returns) < sum(cheap.returns)
    assert dear.metrics["total_costs"] > cheap.metrics["total_costs"]


def test_flat_strategy_never_pays_costs_and_never_moves():
    res = backtest.run(strategy.get("flat"), synthetic_bars(n=300),
                       backtest.Costs(10, 10, 5, 100))
    assert res.metrics["total_costs"] == 0.0
    assert all(r == 0.0 for r in res.returns)


def test_buy_hold_tracks_the_asset():
    bars = rising_bars(100, 0.01)
    res = backtest.run(strategy.get("buy_hold"), bars, backtest.Costs(0, 0, 0, 0))
    asset = bars.close[-1] / bars.close[0] - 1
    strat = 1.0
    for r in res.returns:
        strat *= (1 + r)
    # One bar of entry lag is the only difference.
    assert strat - 1 == pytest.approx(asset, rel=0.02)


def test_weights_are_clamped_to_unit_range():
    s = strategy.Strategy("greedy", lambda b: [5.0] * len(b))
    res = backtest.run(s, synthetic_bars(n=100), max_leverage=1.0)
    assert max(abs(w) for w in res.weights) <= 1.0


def test_warmup_drops_leading_bars():
    bars = synthetic_bars(n=300)
    full = backtest.run(strategy.get("sma_cross"), bars)
    warm = backtest.run(strategy.get("sma_cross"), bars, warmup=100)
    assert len(warm.returns) == len(full.returns) - 100
    assert warm.returns == full.returns[100:]


def test_strategy_returning_wrong_length_is_rejected():
    s = strategy.Strategy("short", lambda b: [0.0] * (len(b) - 1))
    with pytest.raises(ValueError):
        backtest.run(s, synthetic_bars(n=50))


def test_short_borrow_is_charged():
    bars = flat_bars(100)
    s = strategy.Strategy("short", lambda b: [-1.0] * len(b))
    res = backtest.run(s, bars, backtest.Costs(0, 0, 0, borrow_bps_annual=1000))
    assert sum(res.costs_paid) > 0
    assert sum(res.returns) < 0
