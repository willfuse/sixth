"""Performance statistics, in pure stdlib.

The article's discipline list is what this module implements: Deflated Sharpe so
"a strategy that got lucky across 4,000 combinations cannot pass itself off as an
edge", plus Monte Carlo across resampled paths.

References
----------
Bailey & Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier" (PSR).
Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio".
Politis & Romano (1994), "The Stationary Bootstrap".
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence

EULER_MASCHERONI = 0.5772156649015329


# --------------------------------------------------------------------------
# normal distribution helpers (stdlib only)
# --------------------------------------------------------------------------
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Inverse normal CDF (Acklam's rational approximation, |eps| < 1.15e-9)."""
    if not 0.0 < p < 1.0:
        raise ValueError("norm_ppf domain is (0, 1), got %r" % (p,))
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        return num / den
    if p > p_high:
        q = math.sqrt(-2 * math.log(1 - p))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        return -num / den
    q = p - 0.5
    r = q * q
    num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    den = (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    return num / den


# --------------------------------------------------------------------------
# moments
# --------------------------------------------------------------------------
def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def stdev(xs: Sequence[float], ddof: int = 1) -> float:
    """Sample standard deviation, with a guard for numerically-zero variance.

    A constant series does not always sum to exactly zero variance: the mean
    carries rounding error, so the squared deviations land at ~1e-36 instead of
    0 on some platforms. Left alone that turns a flat return stream into a
    Sharpe of 1e16. Anything below floating-point noise for the data's own scale
    is treated as the zero it is.
    """
    n = len(xs)
    if n - ddof <= 0:
        return 0.0
    m = mean(xs)
    ss = sum((x - m) ** 2 for x in xs)
    scale = max(abs(m), max((abs(x) for x in xs), default=0.0))
    if ss <= (1e-12 * scale) ** 2 * n:
        return 0.0
    return math.sqrt(ss / (n - ddof))


def skewness(xs: Sequence[float]) -> float:
    n, s = len(xs), stdev(xs, ddof=0)
    if n < 3 or s == 0:
        return 0.0
    m = mean(xs)
    return sum(((x - m) / s) ** 3 for x in xs) / n


def kurtosis(xs: Sequence[float]) -> float:
    """Non-excess (normal == 3.0)."""
    n, s = len(xs), stdev(xs, ddof=0)
    if n < 4 or s == 0:
        return 3.0
    m = mean(xs)
    return sum(((x - m) / s) ** 4 for x in xs) / n


# --------------------------------------------------------------------------
# headline metrics
# --------------------------------------------------------------------------
def sharpe(returns: Sequence[float], periods_per_year: int = 252,
           rf_per_period: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    ex = [r - rf_per_period for r in returns]
    s = stdev(ex)
    if s == 0:
        return 0.0
    return mean(ex) / s * math.sqrt(periods_per_year)


def sortino(returns: Sequence[float], periods_per_year: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    downside = [r for r in returns if r < 0]
    if not downside:
        return float("inf") if mean(returns) > 0 else 0.0
    dd = math.sqrt(sum(r * r for r in downside) / len(returns))
    if dd == 0:
        return 0.0
    return mean(returns) / dd * math.sqrt(periods_per_year)


def equity_curve(returns: Sequence[float], start: float = 1.0) -> List[float]:
    eq, v = [], start
    for r in returns:
        v *= (1.0 + r)
        eq.append(v)
    return eq


def max_drawdown(returns: Sequence[float]) -> float:
    """Worst peak-to-trough as a positive fraction."""
    peak, worst, v = 1.0, 0.0, 1.0
    for r in returns:
        v *= (1.0 + r)
        peak = max(peak, v)
        worst = max(worst, (peak - v) / peak)
    return worst


def drawdown_series(returns: Sequence[float]) -> List[float]:
    out, peak, v = [], 1.0, 1.0
    for r in returns:
        v *= (1.0 + r)
        peak = max(peak, v)
        out.append((peak - v) / peak)
    return out


def cagr(returns: Sequence[float], periods_per_year: int = 252) -> float:
    if not returns:
        return 0.0
    total = 1.0
    for r in returns:
        total *= (1.0 + r)
    years = len(returns) / periods_per_year
    if years <= 0 or total <= 0:
        return -1.0
    return total ** (1.0 / years) - 1.0


def hit_rate(returns: Sequence[float]) -> float:
    nz = [r for r in returns if r != 0]
    return (sum(1 for r in nz if r > 0) / len(nz)) if nz else 0.0


def profit_factor(returns: Sequence[float]) -> float:
    gains = sum(r for r in returns if r > 0)
    losses = -sum(r for r in returns if r < 0)
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


# --------------------------------------------------------------------------
# Probabilistic / Deflated Sharpe
# --------------------------------------------------------------------------
def probabilistic_sharpe(returns: Sequence[float], benchmark_sr: float = 0.0,
                         periods_per_year: int = 252) -> float:
    """P(true Sharpe > benchmark_sr), correcting for skew, kurtosis and length.

    benchmark_sr is annualised, as is the observed Sharpe.
    """
    n = len(returns)
    if n < 3:
        return 0.0
    sr_ann = sharpe(returns, periods_per_year)
    scale = math.sqrt(periods_per_year)
    sr, sr_star = sr_ann / scale, benchmark_sr / scale  # per-period
    g3, g4 = skewness(returns), kurtosis(returns)
    denom = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr * sr
    if denom <= 0:
        return 0.0
    z = (sr - sr_star) * math.sqrt(n - 1) / math.sqrt(denom)
    return norm_cdf(z)


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """E[max Sharpe] over n_trials independent strategies with zero true edge.

    This is the bar an overfit search has to clear. sr_variance is the variance
    of the per-period Sharpes actually tried.
    """
    if n_trials < 2 or sr_variance <= 0:
        return 0.0
    g = EULER_MASCHERONI
    a = norm_ppf(1.0 - 1.0 / n_trials)
    b = norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(sr_variance) * ((1.0 - g) * a + g * b)


def deflated_sharpe(returns: Sequence[float], n_trials: int,
                    trial_sharpes: Optional[Sequence[float]] = None,
                    periods_per_year: int = 252) -> float:
    """Probability the strategy's edge survives the selection bias of the search.

    n_trials is how many variants you actually looked at -- not how many you are
    reporting. The hypothesis graph knows this number, which is the whole point
    of keeping one: honest trial counts are the input nobody has.
    """
    if len(returns) < 3 or n_trials < 1:
        return 0.0
    scale = math.sqrt(periods_per_year)
    if trial_sharpes and len(trial_sharpes) > 1:
        var = stdev([s / scale for s in trial_sharpes]) ** 2
    else:
        # No recorded population: assume trials are ~N(0, 1/n) under the null.
        var = 1.0 / max(len(returns) - 1, 1)
    sr0_ann = expected_max_sharpe(n_trials, var) * scale
    return probabilistic_sharpe(returns, benchmark_sr=sr0_ann,
                                periods_per_year=periods_per_year)


def min_track_record_length(returns: Sequence[float], target_sr: float = 0.0,
                            confidence: float = 0.95,
                            periods_per_year: int = 252) -> float:
    """Periods needed before the Sharpe is distinguishable from target_sr."""
    scale = math.sqrt(periods_per_year)
    sr = sharpe(returns, periods_per_year) / scale
    srs = target_sr / scale
    if sr <= srs:
        return float("inf")
    g3, g4 = skewness(returns), kurtosis(returns)
    num = (1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr * sr)
    return 1.0 + num * (norm_ppf(confidence) / (sr - srs)) ** 2


# --------------------------------------------------------------------------
# Monte Carlo
# --------------------------------------------------------------------------
def stationary_bootstrap(returns: Sequence[float], n_paths: int = 1000,
                         mean_block: int = 10, seed: int = 0) -> List[List[float]]:
    """Politis-Romano stationary bootstrap: resamples paths while preserving
    short-range autocorrelation (which iid resampling destroys, flattering
    trend strategies)."""
    n = len(returns)
    if n == 0:
        return []
    rng = random.Random(seed)
    p = 1.0 / max(mean_block, 1)
    paths = []
    for _ in range(n_paths):
        path, i = [], rng.randrange(n)
        for _ in range(n):
            path.append(returns[i])
            i = rng.randrange(n) if rng.random() < p else (i + 1) % n
        paths.append(path)
    return paths


def percentile(xs: Sequence[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


def monte_carlo(returns: Sequence[float], n_paths: int = 1000,
                mean_block: int = 10, seed: int = 0,
                periods_per_year: int = 252) -> Dict[str, float]:
    """Distribution of outcomes across resampled paths of the same returns."""
    paths = stationary_bootstrap(returns, n_paths, mean_block, seed)
    if not paths:
        return {}
    sharpes = [sharpe(p, periods_per_year) for p in paths]
    dds = [max_drawdown(p) for p in paths]
    totals = [math.prod([1 + r for r in p]) - 1 for p in paths]
    return {
        "mc_paths": float(n_paths),
        "mc_sharpe_p05": percentile(sharpes, 0.05),
        "mc_sharpe_p50": percentile(sharpes, 0.50),
        "mc_sharpe_p95": percentile(sharpes, 0.95),
        "mc_p_sharpe_gt_0": sum(1 for s in sharpes if s > 0) / len(sharpes),
        "mc_maxdd_p95": percentile(dds, 0.95),
        "mc_return_p05": percentile(totals, 0.05),
        "mc_p_profitable": sum(1 for t in totals if t > 0) / len(totals),
    }


def summarize(returns: Sequence[float], periods_per_year: int = 252) -> Dict[str, float]:
    return {
        "n_periods": float(len(returns)),
        "cagr": cagr(returns, periods_per_year),
        "sharpe": sharpe(returns, periods_per_year),
        "sortino": sortino(returns, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "hit_rate": hit_rate(returns),
        "profit_factor": profit_factor(returns),
        "skew": skewness(returns),
        "kurtosis": kurtosis(returns),
        "volatility": stdev(returns) * math.sqrt(periods_per_year),
        "psr": probabilistic_sharpe(returns, 0.0, periods_per_year),
    }
