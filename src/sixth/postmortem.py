"""Post-mortem: read what actually happened, and why.

Not a performance summary -- an autopsy. The output is meant to answer one
question: where did this bleed, and was it the idea, the costs, the sizing, or
one regime it should never have traded in?

Everything here is deterministic and computed from the return stream, so the same
run always produces the same findings. That matters because these findings become
graph rows, and a graph built from unstable observations is worse than no graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .backtest import BacktestResult
from .data import Bars
from .prereg import CheckResult, Expectation, evaluate, passing_regimes
from .regimes import label_regimes, regime_breakdown
from .stats import drawdown_series, mean, percentile, sharpe, stdev


@dataclass
class Episode:
    """A drawdown from peak to recovery (or to the end of the sample)."""

    start_date: str
    trough_date: str
    end_date: Optional[str]
    depth: float
    length: int
    recovered: bool
    regime: str = ""


@dataclass
class PostMortem:
    strategy: str
    metrics: Dict[str, float]
    regimes: Dict[str, Dict[str, float]]
    episodes: List[Episode] = field(default_factory=list)
    findings: List[str] = field(default_factory=list)
    verdict: str = ""
    checks: List[CheckResult] = field(default_factory=list)
    passing_regimes: List[str] = field(default_factory=list)
    failing_regimes: List[str] = field(default_factory=list)
    cost_drag: float = 0.0
    concentration: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy, "verdict": self.verdict,
            "metrics": self.metrics, "regimes": self.regimes,
            "cost_drag": self.cost_drag, "concentration": self.concentration,
            "passing_regimes": self.passing_regimes,
            "failing_regimes": self.failing_regimes,
            "findings": self.findings,
            "episodes": [vars(e) for e in self.episodes],
            "checks": [vars(c) for c in self.checks],
        }


def analyse(result: BacktestResult, bars: Bars,
            expectation: Optional[Expectation] = None,
            periods_per_year: int = 252, n_episodes: int = 5) -> PostMortem:
    """Produce the autopsy."""
    returns = result.returns
    labels = label_regimes(bars)[-len(returns):]
    breakdown = regime_breakdown(returns, labels, periods_per_year)

    pm = PostMortem(strategy=result.strategy, metrics=dict(result.metrics),
                    regimes=breakdown)
    pm.episodes = drawdown_episodes(returns, result.dates, labels, n_episodes)
    pm.cost_drag = _cost_drag(result)
    pm.concentration = _concentration(returns)
    pm.failing_regimes = sorted(k for k, v in breakdown.items() if v["sharpe"] < 0)

    if expectation is not None:
        pm.verdict, pm.checks = evaluate(expectation, result.metrics, breakdown)
        pm.passing_regimes = passing_regimes(expectation, breakdown)
    pm.findings = _findings(pm, result, breakdown)
    return pm


def drawdown_episodes(returns: Sequence[float], dates: Sequence[str],
                      labels: Sequence[str], top: int = 5) -> List[Episode]:
    """Every peak-to-recovery drawdown, deepest first."""
    dd = drawdown_series(returns)
    episodes: List[Episode] = []
    i, n = 0, len(dd)
    while i < n:
        if dd[i] <= 0:
            i += 1
            continue
        start = i
        trough = i
        while i < n and dd[i] > 0:
            if dd[i] > dd[trough]:
                trough = i
            i += 1
        recovered = i < n
        end = i - 1
        episodes.append(Episode(
            start_date=dates[start] if start < len(dates) else "",
            trough_date=dates[trough] if trough < len(dates) else "",
            end_date=dates[end] if recovered and end < len(dates) else None,
            depth=dd[trough], length=end - start + 1, recovered=recovered,
            regime=labels[trough] if trough < len(labels) else "",
        ))
    episodes.sort(key=lambda e: -e.depth)
    return episodes[:top]


def _cost_drag(result: BacktestResult) -> float:
    """Fraction of the gross edge consumed by frictions.

    >1.0 means the idea made money and the execution took all of it -- a
    different failure from "the idea was wrong", and one that a plain Sharpe
    number hides completely.
    """
    gross = sum(result.gross_returns)
    paid = sum(result.costs_paid)
    if gross <= 0:
        return float("inf") if paid > 0 else 0.0
    return paid / gross


def _concentration(returns: Sequence[float]) -> float:
    """Share of total profit produced by the best 5% of bars.

    Near 1.0 means the track record is a handful of lucky days wearing a strategy
    costume.
    """
    gains = sorted((r for r in returns if r > 0), reverse=True)
    if not gains:
        return 0.0
    k = max(1, int(len(returns) * 0.05))
    return sum(gains[:k]) / sum(gains)


def _findings(pm: PostMortem, result: BacktestResult,
              breakdown: Dict[str, Dict[str, float]]) -> List[str]:
    """Plain-language diagnoses. These become lessons, so each one names the
    mechanism, not just the symptom."""
    out: List[str] = []
    m = result.metrics

    if pm.cost_drag == float("inf"):
        out.append(
            f"Gross returns were negative before costs, at "
            f"{m.get('turnover_annual', 0):.1f}x annual turnover: frictions are not the "
            f"problem, the signal is.")
    elif pm.cost_drag > 1.0:
        out.append(
            f"Costs consumed {pm.cost_drag:.0%} of the gross edge. The idea worked and the "
            f"execution ate it -- turnover is {m.get('turnover_annual', 0):.1f}x a year.")
    elif pm.cost_drag > 0.5:
        out.append(
            f"Costs took {pm.cost_drag:.0%} of the gross edge; the strategy needs cheap fills "
            f"to stay viable.")

    if pm.concentration > 0.6:
        out.append(
            f"{pm.concentration:.0%} of all gains came from the best 5% of bars. This is a "
            f"few lucky days, not a distribution.")

    if breakdown:
        worst = min(breakdown, key=lambda k: breakdown[k]["sharpe"])
        best = max(breakdown, key=lambda k: breakdown[k]["sharpe"])
        if breakdown[worst]["sharpe"] < -0.3:
            out.append(
                f"Bleeds in {worst}: Sharpe {breakdown[worst]['sharpe']:.2f} across "
                f"{int(breakdown[worst]['n'])} bars ({breakdown[worst]['share']:.0%} of the "
                f"sample). A regime filter that sits out {worst} is the obvious next variant.")
        if breakdown[best]["sharpe"] > 0.5 and breakdown[best]["share"] < 0.35:
            out.append(
                f"Almost all the edge lives in {best} ({breakdown[best]['share']:.0%} of bars). "
                f"Treat this as regime-conditional, not general.")

    if m.get("max_drawdown", 0) > 0.4:
        out.append(f"Max drawdown {m['max_drawdown']:.0%} -- unsizeable without a risk overlay.")

    if m.get("time_in_market", 1.0) < 0.15:
        out.append(
            f"In the market only {m['time_in_market']:.0%} of the time; the sample of actual "
            f"decisions is far smaller than the bar count suggests.")

    if pm.episodes:
        e = pm.episodes[0]
        state = "still underwater at the end of the sample" if not e.recovered \
            else f"took {e.length} bars to recover"
        out.append(
            f"Worst episode: -{e.depth:.0%} from {e.start_date}, trough {e.trough_date} "
            f"in {e.regime}, {state}.")

    if m.get("gross_sharpe", 0) > 0.5 and m.get("sharpe", 0) < 0.2:
        out.append(
            "Gross Sharpe is respectable but net is not: this is an execution problem, and "
            "the fix is turnover reduction, not a new signal.")

    if not out:
        out.append("No structural weakness found by the standard checks.")
    return out


def render(pm: PostMortem) -> str:
    """Human-readable autopsy."""
    L: List[str] = []
    L.append(f"POST-MORTEM: {pm.strategy}")
    L.append("=" * 60)
    if pm.verdict:
        L.append(f"Verdict against sealed expectation: {pm.verdict.upper()}")
        for c in pm.checks:
            L.append("  " + c.describe())
        if pm.passing_regimes:
            L.append(f"  Conditions held in: {', '.join(pm.passing_regimes)}")
        L.append("")

    m = pm.metrics
    L.append("HEADLINE")
    for k in ("sharpe", "gross_sharpe", "cagr", "max_drawdown", "hit_rate",
              "turnover_annual", "time_in_market", "psr"):
        if k in m:
            L.append(f"  {k:<18} {m[k]:>10.4f}")
    L.append(f"  {'cost_drag':<18} {pm.cost_drag:>10.4f}")
    L.append(f"  {'concentration':<18} {pm.concentration:>10.4f}")
    L.append("")

    if pm.regimes:
        L.append("BY REGIME")
        L.append(f"  {'regime':<18}{'bars':>6}{'share':>8}{'sharpe':>9}{'total':>9}")
        for lab, r in sorted(pm.regimes.items(), key=lambda kv: kv[1]["sharpe"]):
            L.append(f"  {lab:<18}{int(r['n']):>6}{r['share']:>8.1%}"
                     f"{r['sharpe']:>9.2f}{r['total_return']:>9.1%}")
        L.append("")

    if pm.episodes:
        L.append("WORST DRAWDOWNS")
        for e in pm.episodes:
            tail = "recovered" if e.recovered else "NOT RECOVERED"
            L.append(f"  -{e.depth:>6.1%}  {e.start_date} -> {e.trough_date}  "
                     f"{e.length:>4} bars  {e.regime:<16} {tail}")
        L.append("")

    L.append("FINDINGS")
    for f in pm.findings:
        L.append(f"  - {f}")
    return "\n".join(L)
