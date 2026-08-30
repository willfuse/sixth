import pytest

from sixth.live import (KillSwitchTripped, Order, PaperBroker, RiskGate,
                        RiskLimits)


def test_position_is_clamped_not_refused():
    g = RiskGate(RiskLimits(max_position=0.5, max_order_notional=1.0))
    assert g.check(Order("AAPL", 5.0)) == 0.5


def test_order_size_is_stepped():
    g = RiskGate(RiskLimits(max_position=1.0, max_order_notional=0.25))
    assert g.check(Order("AAPL", 1.0), current_weight=0.0) == 0.25


def test_shorts_are_refused_when_disallowed():
    g = RiskGate(RiskLimits(allow_shorts=False))
    assert g.check(Order("AAPL", -1.0)) == 0.0


def test_shorts_allowed_when_enabled():
    g = RiskGate(RiskLimits(allow_shorts=True, max_position=1.0,
                            max_order_notional=1.0))
    assert g.check(Order("AAPL", -1.0)) == -1.0


def test_gross_exposure_limits_the_whole_book():
    g = RiskGate(RiskLimits(max_gross_exposure=1.0, max_position=1.0,
                            max_order_notional=1.0))
    assert g.check(Order("MSFT", 0.8), book={"AAPL": 0.7}) == pytest.approx(0.3)


def test_drawdown_trips_the_kill_switch():
    g = RiskGate(RiskLimits(max_drawdown=0.10), equity=100.0)
    g.mark(100.0, day="d1")
    g.mark(85.0, day="d1")
    assert g.halted
    with pytest.raises(KillSwitchTripped):
        g.check(Order("AAPL", 0.1))


def test_daily_loss_trips_the_kill_switch():
    g = RiskGate(RiskLimits(max_daily_loss=0.02, max_drawdown=0.99), equity=100.0)
    g.mark(100.0, day="d1")
    g.mark(97.0, day="d1")
    assert g.halted and "daily loss" in g.halt_reason


def test_new_day_resets_the_daily_baseline():
    g = RiskGate(RiskLimits(max_daily_loss=0.02, max_drawdown=0.99), equity=100.0)
    g.mark(100.0, day="d1")
    g.mark(99.0, day="d1")
    g.mark(99.0, day="d2")
    g.mark(98.5, day="d2")     # -0.5% on the new day, not -1.5% cumulative
    assert not g.halted


def test_order_count_limit_halts():
    g = RiskGate(RiskLimits(max_orders_per_day=2, max_order_notional=1.0))
    g.check(Order("A", 0.1))
    g.check(Order("A", 0.1))
    with pytest.raises(KillSwitchTripped):
        g.check(Order("A", 0.1))


def test_a_halted_gate_stays_halted():
    """No sequence of later calls un-trips it. That is the point of a switch."""
    g = RiskGate(RiskLimits(max_drawdown=0.05), equity=100.0)
    g.mark(100.0, day="d1")
    g.mark(90.0, day="d1")
    g.mark(120.0, day="d2")        # recovery does not resume trading
    assert g.halted
    with pytest.raises(KillSwitchTripped):
        g.check(Order("A", 0.1))


def test_clamps_are_journalled_with_the_rule_that_fired():
    g = RiskGate(RiskLimits(max_position=0.5, max_order_notional=1.0))
    g.check(Order("AAPL", 5.0))
    rules = [e.get("rule") for e in g.journal]
    assert "max_position" in rules


def test_journal_is_written_as_jsonl(tmp_path):
    import json
    g = RiskGate(RiskLimits(max_position=0.5, max_order_notional=1.0))
    g.check(Order("AAPL", 5.0, reason="signal fired"))
    path = tmp_path / "journal.jsonl"
    g.write_journal(str(path))
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    assert rows and rows[0]["event"] == "CLAMP"


def test_paper_broker_charges_costs_and_tracks_positions():
    b = PaperBroker(equity=100_000.0, cost_bps=10.0)
    b.submit(Order("AAPL", 1.0), 1.0)
    assert b.positions()["AAPL"] == 1.0
    assert b.equity() < 100_000.0


def test_paper_broker_marks_the_book_forward():
    b = PaperBroker(equity=100_000.0, cost_bps=0.0)
    b.submit(Order("AAPL", 1.0), 1.0)
    b.apply_return("AAPL", 0.10)
    assert b.equity() == pytest.approx(110_000.0)


def test_base_adapter_refuses_to_pretend_it_can_trade():
    from sixth.live import BrokerAdapter
    with pytest.raises(NotImplementedError):
        BrokerAdapter().positions()
    with pytest.raises(NotImplementedError):
        BrokerAdapter().equity()


def test_the_documented_example_actually_trips():
    """The README and the guide both print this snippet. It has to work.

    An earlier version omitted `equity=`, so the gate defaulted to 1.0 and
    marking 85,000 read as a gain rather than a 15% drawdown -- the example
    never tripped.
    """
    gate = RiskGate(RiskLimits(max_position=0.5, max_drawdown=0.10,
                               max_daily_loss=0.02, allow_shorts=False),
                    equity=100_000)
    assert gate.check(Order("AAPL", 5.0)) == 0.25
    assert gate.check(Order("AAPL", -1.0)) == 0.0
    gate.mark(equity=85_000)
    with pytest.raises(KillSwitchTripped):
        gate.check(Order("AAPL", 0.1))
