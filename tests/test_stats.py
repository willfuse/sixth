import math
import random

import pytest

from sixth import stats


def test_norm_ppf_inverts_norm_cdf():
    for p in (0.001, 0.025, 0.1, 0.5, 0.9, 0.975, 0.999):
        assert stats.norm_cdf(stats.norm_ppf(p)) == pytest.approx(p, abs=1e-6)


def test_norm_ppf_known_values():
    assert stats.norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-5)
    assert stats.norm_ppf(0.5) == pytest.approx(0.0, abs=1e-12)


def test_norm_ppf_rejects_domain():
    for bad in (0.0, 1.0, -0.1, 1.2):
        with pytest.raises(ValueError):
            stats.norm_ppf(bad)


def test_sharpe_scales_with_annualisation():
    r = [0.001] * 252
    assert stats.sharpe(r) == 0.0  # zero variance -> zero, not a division error
    rng = random.Random(0)
    noisy = [rng.gauss(0.001, 0.01) for _ in range(2520)]
    daily = stats.sharpe(noisy, 1)
    assert stats.sharpe(noisy, 252) == pytest.approx(daily * math.sqrt(252), rel=1e-9)


def test_max_drawdown_is_positive_fraction():
    assert stats.max_drawdown([0.5, -0.5]) == pytest.approx(0.5)
    assert stats.max_drawdown([0.1, 0.1]) == 0.0


def test_deflated_sharpe_falls_as_trials_rise():
    rng = random.Random(1)
    r = [rng.gauss(0.0005, 0.01) for _ in range(1260)]
    one = stats.deflated_sharpe(r, 1)
    many = stats.deflated_sharpe(r, 4000)
    assert one > 0.5
    assert many < 0.1
    assert many < one


def test_deflated_sharpe_is_monotone_in_trials():
    rng = random.Random(2)
    r = [rng.gauss(0.0008, 0.01) for _ in range(1260)]
    vals = [stats.deflated_sharpe(r, n) for n in (1, 10, 100, 1000, 10000)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))


def test_psr_rises_with_sample_length():
    rng = random.Random(3)
    short = [rng.gauss(0.0006, 0.01) for _ in range(120)]
    long = short + [rng.gauss(0.0006, 0.01) for _ in range(2400)]
    assert stats.probabilistic_sharpe(long) > stats.probabilistic_sharpe(short)


def test_expected_max_sharpe_grows_with_trials():
    a = stats.expected_max_sharpe(10, 0.01)
    b = stats.expected_max_sharpe(1000, 0.01)
    assert 0 < a < b


def test_monte_carlo_reports_a_distribution():
    rng = random.Random(4)
    r = [rng.gauss(0.0005, 0.01) for _ in range(756)]
    mc = stats.monte_carlo(r, n_paths=200, seed=1)
    assert mc["mc_sharpe_p05"] <= mc["mc_sharpe_p50"] <= mc["mc_sharpe_p95"]
    assert 0.0 <= mc["mc_p_profitable"] <= 1.0


def test_stationary_bootstrap_is_deterministic_given_seed():
    r = [0.01, -0.02, 0.03, 0.0, -0.01] * 20
    a = stats.stationary_bootstrap(r, 5, 10, seed=42)
    b = stats.stationary_bootstrap(r, 5, 10, seed=42)
    assert a == b


def test_summarize_handles_empty_and_flat():
    assert stats.summarize([])["n_periods"] == 0
    flat = stats.summarize([0.0] * 100)
    assert flat["sharpe"] == 0.0
    assert flat["max_drawdown"] == 0.0
