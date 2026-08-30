import pytest

from sixth.prereg import (CONFIRMED, INCONCLUSIVE, REFUTED, REGIME_CONDITIONAL,
                          Expectation, evaluate, passing_regimes)


def test_parse_all_operators():
    e = Expectation.parse(["sharpe >= 0.8", "max_drawdown <= 0.25", "trades > 10"])
    assert e.must["sharpe"] == {">=": 0.8}
    assert e.must["max_drawdown"] == {"<=": 0.25}
    assert e.must["trades"] == {">": 10.0}


def test_parse_rejects_malformed():
    with pytest.raises(ValueError):
        Expectation.parse(["sharpe is good"])
    with pytest.raises(ValueError):
        Expectation.parse(["sharpe >= lots"])


def test_seal_is_stable_and_detects_change():
    e = Expectation.parse(["sharpe >= 0.8"], rationale="because")
    ts, h = e.seal()
    assert e.verify(ts, h)
    assert Expectation.parse(["sharpe >= 0.1"], rationale="because").verify(ts, h) is False


def test_seal_ignores_dict_ordering():
    a = Expectation.parse(["sharpe >= 0.8", "max_drawdown <= 0.2"])
    b = Expectation.parse(["max_drawdown <= 0.2", "sharpe >= 0.8"])
    ts, h = a.seal()
    assert b.verify(ts, h)


def test_roundtrip_json():
    e = Expectation.parse(["sharpe >= 0.8"], rationale="why", regimes=["up/calm"],
                          should=["max_drawdown <= 0.3"])
    back = Expectation.from_json(e.to_json())
    assert back.must == e.must
    assert back.should == e.should
    assert back.rationale == e.rationale
    assert back.expected_regimes == e.expected_regimes


def test_all_must_pass_is_confirmed():
    e = Expectation.parse(["sharpe >= 0.8", "max_drawdown <= 0.25"])
    v, checks = evaluate(e, {"sharpe": 1.0, "max_drawdown": 0.1})
    assert v == CONFIRMED
    assert all(c.passed for c in checks)


def test_failing_should_downgrades_to_inconclusive():
    e = Expectation.parse(["sharpe >= 0.5"], should=["max_drawdown <= 0.1"])
    v, _ = evaluate(e, {"sharpe": 0.9, "max_drawdown": 0.4})
    assert v == INCONCLUSIVE


def test_failure_everywhere_is_refuted():
    e = Expectation.parse(["sharpe >= 0.8"])
    v, _ = evaluate(e, {"sharpe": 0.1}, {"up/calm": {"sharpe": 0.2}})
    assert v == REFUTED


def test_failure_that_holds_somewhere_is_regime_conditional():
    e = Expectation.parse(["sharpe >= 0.8"])
    v, _ = evaluate(e, {"sharpe": 0.1},
                    {"up/calm": {"sharpe": 1.4}, "down/normal": {"sharpe": -0.9}})
    assert v == REGIME_CONDITIONAL


def test_missing_metric_fails_rather_than_passing_silently():
    e = Expectation.parse(["sharpe >= 0.8"])
    v, checks = evaluate(e, {"cagr": 0.2})
    assert v == REFUTED
    assert checks[0].actual is None and checks[0].passed is False


def test_no_must_conditions_is_inconclusive():
    v, _ = evaluate(Expectation(), {"sharpe": 5.0})
    assert v == INCONCLUSIVE


def test_passing_regimes_lists_only_full_matches():
    e = Expectation.parse(["sharpe >= 0.8", "max_drawdown <= 0.2"])
    got = passing_regimes(e, {
        "up/calm": {"sharpe": 1.2, "max_drawdown": 0.1},     # both
        "up/normal": {"sharpe": 1.2, "max_drawdown": 0.5},   # dd fails
        "down/x": {"sharpe": -1.0, "max_drawdown": 0.1},     # sharpe fails
    })
    assert got == ["up/calm"]
