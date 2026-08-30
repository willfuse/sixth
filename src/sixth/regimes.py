"""Regime labelling.

The graph's core claim is not "this failed" but "this failed *here*". A refuted
hypothesis with no regime attached is nearly worthless -- it might have been a
perfectly good idea tested in the one environment that kills it. So every result
gets sliced by regime before it is written down.

Labels come from two observable axes, both computed causally (trailing windows
only, no lookahead): trend direction and volatility level.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from .data import Bars
from .stats import percentile, stdev

TREND_LABELS = ("up", "flat", "down")
VOL_LABELS = ("calm", "normal", "stressed")


def rolling_mean(xs: Sequence[float], window: int) -> List[float]:
    """Trailing mean; entries before the window is full are None-free copies of
    the partial mean, so downstream code never has to special-case the warmup."""
    out, acc = [], 0.0
    for i, x in enumerate(xs):
        acc += x
        if i >= window:
            acc -= xs[i - window]
        out.append(acc / min(i + 1, window))
    return out


def rolling_std(xs: Sequence[float], window: int) -> List[float]:
    out = []
    for i in range(len(xs)):
        lo = max(0, i - window + 1)
        out.append(stdev(xs[lo:i + 1]) if i > 0 else 0.0)
    return out


def label_regimes(bars: Bars, trend_window: int = 100, vol_window: int = 40,
                  trend_threshold: float = 0.02) -> List[str]:
    """Per-bar regime label, e.g. "up/calm". Causal: bar i uses bars <= i only."""
    close = bars.close
    rets = bars.returns()
    ma = rolling_mean(close, trend_window)
    vol = rolling_std(rets, vol_window)

    # Volatility cutoffs from the full sample. This is a labelling convention,
    # not a signal, so it does not leak into strategy decisions -- but keep it
    # out of any feature you actually trade on.
    ref = [v for v in vol[vol_window:] if v > 0]
    calm_cut = percentile(ref, 0.33) if ref else 0.0
    stress_cut = percentile(ref, 0.80) if ref else 0.0

    labels = []
    for i in range(len(close)):
        dev = (close[i] - ma[i]) / ma[i] if ma[i] else 0.0
        if dev > trend_threshold:
            trend = "up"
        elif dev < -trend_threshold:
            trend = "down"
        else:
            trend = "flat"
        v = vol[i]
        if v <= calm_cut:
            vlab = "calm"
        elif v >= stress_cut:
            vlab = "stressed"
        else:
            vlab = "normal"
        labels.append(f"{trend}/{vlab}")
    return labels


def regime_breakdown(returns: Sequence[float], labels: Sequence[str],
                     periods_per_year: int = 252,
                     min_periods: int = 20) -> Dict[str, Dict[str, float]]:
    """Split a return stream by regime. This is the shape of what gets stored on
    the hypothesis, and what makes `confirmed_only_in()` answerable later."""
    from .stats import max_drawdown, mean, sharpe

    buckets: Dict[str, List[float]] = {}
    for r, lab in zip(returns, labels):
        buckets.setdefault(lab, []).append(r)

    out: Dict[str, Dict[str, float]] = {}
    for lab, rs in sorted(buckets.items()):
        if len(rs) < min_periods:
            continue
        total = 1.0
        for r in rs:
            total *= (1.0 + r)
        out[lab] = {
            "n": float(len(rs)),
            "share": len(rs) / len(returns) if returns else 0.0,
            "mean_return": mean(rs),
            "total_return": total - 1.0,
            "sharpe": sharpe(rs, periods_per_year),
            "max_drawdown": max_drawdown(rs),
        }
    return out


def worst_windows(bars: Bars, k: int = 3, window: int = 60) -> List[Dict[str, object]]:
    """The k worst non-overlapping drawdown windows in the data.

    The breaker replays a strategy through exactly these -- "the ten worst
    historical regimes for that asset class" from the article, computed rather
    than hand-picked.
    """
    close = bars.close
    n = len(close)
    if n < window + 1:
        return []
    scored = []
    for i in range(0, n - window):
        a, b = close[i], close[i + window]
        scored.append(((b - a) / a, i))
    scored.sort(key=lambda t: t[0])

    chosen: List[Dict[str, object]] = []
    used: List[int] = []
    for ret, i in scored:
        if any(abs(i - j) < window for j in used):
            continue
        used.append(i)
        chosen.append({
            "start": i, "end": i + window,
            "start_date": bars.dates[i], "end_date": bars.dates[i + window],
            "return": ret,
        })
        if len(chosen) >= k:
            break
    return chosen
