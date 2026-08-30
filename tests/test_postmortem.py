import pytest

from sixth.backtest import Costs, run
from sixth.data import Bars
from sixth.postmortem import analyse, drawdown_episodes, render
from sixth.prereg import Expectation
from sixth.regimes import label_regimes, regime_breakdown, worst_windows
from sixth.strategy import get


def test_drawdown_episodes_are_ordered_by_depth():
    returns = [0.1, -0.3, 0.4, -0.1, 0.2]
    dates = [f"d{i}" for i in range(len(returns))]
    eps = drawdown_episodes(returns, dates, ["x"] * len(returns))
    assert eps
    assert eps == sorted(eps, key=lambda e: -e.depth)


def test_unrecovered_drawdown_is_flagged():
    returns = [0.1] + [-0.02] * 20
    dates = [f"d{i}" for i in range(len(returns))]
    eps = drawdown_episodes(returns, dates, ["x"] * len(returns))
    assert eps[0].recovered is False
    assert eps[0].end_date is None


def test_no_drawdown_yields_no_episodes():
    returns = [0.01] * 30
    eps = drawdown_episodes(returns, [f"d{i}" for i in range(30)], ["x"] * 30)
    assert eps == []


def test_cost_drag_is_infinite_when_gross_is_negative(bars):
    """A losing signal is a different diagnosis from an expensive one."""
    res = run(get("random_signal", seed=1), bars, Costs(20, 20, 10, 0))
    pm = analyse(res, bars)
    if sum(res.gross_returns) <= 0:
        assert pm.cost_drag == float("inf")
        assert any("signal is" in f for f in pm.findings)


def test_concentration_detects_a_single_lucky_bar():
    n = 200
    returns = [0.0] * n
    returns[50] = 1.0
    dates = [f"d{i}" for i in range(n)]
    bars = Bars("X", dates, [100.0] * n, [100.0] * n, [100.0] * n, [100.0] * n)

    from sixth.postmortem import _concentration
    assert _concentration(returns) == pytest.approx(1.0)


def test_verdict_and_checks_come_from_the_expectation(bars):
    res = run(get("buy_hold"), bars)
    exp = Expectation.parse(["sharpe >= 99.0"])
    pm = analyse(res, bars, exp)
    assert pm.verdict in ("refuted", "regime_conditional")
    assert pm.checks and not pm.checks[0].passed


def test_regime_breakdown_shares_sum_to_one(bars):
    res = run(get("sma_cross"), bars)
    labels = label_regimes(bars)[-len(res.returns):]
    bd = regime_breakdown(res.returns, labels)
    assert sum(v["share"] for v in bd.values()) == pytest.approx(1.0, abs=0.05)


def test_regime_breakdown_drops_tiny_buckets():
    returns = [0.01] * 100
    labels = ["big"] * 95 + ["tiny"] * 5
    bd = regime_breakdown(returns, labels, min_periods=20)
    assert "big" in bd and "tiny" not in bd


def test_worst_windows_do_not_overlap(bars):
    ws = worst_windows(bars, k=4, window=60)
    starts = sorted(int(w["start"]) for w in ws)
    for a, b in zip(starts, starts[1:]):
        assert b - a >= 60


def test_worst_windows_are_actually_the_worst(bars):
    ws = worst_windows(bars, k=3, window=60)
    assert all(w["return"] < 0 for w in ws)
    assert ws[0]["return"] <= ws[-1]["return"]


def test_render_produces_all_sections(bars):
    res = run(get("sma_cross"), bars)
    text = render(analyse(res, bars, Expectation.parse(["sharpe >= 0.8"])))
    for section in ("POST-MORTEM", "HEADLINE", "BY REGIME", "FINDINGS"):
        assert section in text


def test_findings_are_never_empty(bars):
    pm = analyse(run(get("buy_hold"), bars), bars)
    assert pm.findings
