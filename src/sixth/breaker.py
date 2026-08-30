"""The breaker.

From the article: "a separate breaker agent whose only job is to find the
conditions where the strategy dies, running it at double transaction costs and
against the ten worst historical regimes for that asset class."

Adversarial by construction. It never tries to make a strategy look good; every
probe here is an attempt to kill it. What survives is what gets written into the
graph, and what does not survive is written down too -- the exact cost level, the
exact regime, the exact lag that broke it. That fragility profile is the most
reusable thing a failed test produces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .backtest import Costs, run
from .data import Bars
from .regimes import label_regimes, regime_breakdown, worst_windows
from .stats import percentile, sharpe, stdev
from .strategy import Strategy, get


@dataclass
class Probe:
    name: str
    description: str
    survived: bool
    metric: float
    baseline: float
    detail: Dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        mark = "SURVIVED" if self.survived else "KILLED"
        return f"[{mark}] {self.name}: {self.description} (sharpe {self.metric:.3f} vs base {self.baseline:.3f})"


@dataclass
class FragilityReport:
    strategy: str
    baseline_sharpe: float
    probes: List[Probe] = field(default_factory=list)
    cost_death_bps: Optional[float] = None
    fatal_regimes: List[str] = field(default_factory=list)
    null_p_value: float = 1.0

    @property
    def survived(self) -> bool:
        return all(p.survived for p in self.probes)

    @property
    def failures(self) -> List[Probe]:
        return [p for p in self.probes if not p.survived]

    @property
    def robustness(self) -> float:
        """Share of probes survived. Not a p-value -- a blunt summary."""
        return (sum(1 for p in self.probes if p.survived) / len(self.probes)
                if self.probes else 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy, "baseline_sharpe": self.baseline_sharpe,
            "survived": self.survived, "robustness": self.robustness,
            "cost_death_bps": self.cost_death_bps,
            "fatal_regimes": self.fatal_regimes, "null_p_value": self.null_p_value,
            "probes": [{"name": p.name, "description": p.description,
                        "survived": p.survived, "metric": p.metric,
                        "baseline": p.baseline, "detail": p.detail}
                       for p in self.probes],
        }


def break_it(strategy: Strategy, bars: Bars, costs: Optional[Costs] = None,
             periods_per_year: int = 252, cost_multipliers: Sequence[float] = (2.0, 4.0),
             n_worst_windows: int = 3, window: int = 60,
             null_samples: int = 200, perturb: float = 0.25,
             min_sharpe: float = 0.0) -> FragilityReport:
    """Run the full battery. Every probe is a way the strategy might die."""
    costs = costs or Costs()
    base = run(strategy, bars, costs, periods_per_year)
    base_sharpe = base.metrics["sharpe"]
    report = FragilityReport(strategy.name, base_sharpe)

    _probe_costs(report, strategy, bars, costs, periods_per_year,
                 cost_multipliers, min_sharpe)
    _probe_worst_windows(report, strategy, bars, costs, periods_per_year,
                         n_worst_windows, window, min_sharpe)
    _probe_regimes(report, base.returns, bars, periods_per_year, min_sharpe)
    _probe_lag(report, strategy, bars, costs, periods_per_year, min_sharpe)
    _probe_params(report, strategy, bars, costs, periods_per_year, perturb, min_sharpe)
    _probe_null(report, strategy, base, bars, costs, periods_per_year, null_samples)
    return report


# --------------------------------------------------------------------------
def _probe_costs(rep: FragilityReport, strat: Strategy, bars: Bars, costs: Costs,
                 ppy: int, multipliers: Sequence[float], min_sharpe: float) -> None:
    """Does the edge survive worse fills? And at what cost level does it die?"""
    for mult in multipliers:
        res = run(strat, bars, costs.scaled(mult), ppy)
        s = res.metrics["sharpe"]
        rep.probes.append(Probe(
            f"costs_x{mult:g}", f"transaction costs multiplied by {mult:g}",
            s > min_sharpe, s, rep.baseline_sharpe, {"multiplier": mult}))

    # Bisect for the cost level that zeroes the edge -- a far more useful number
    # than a pass/fail, because it says how much execution quality you can afford.
    if rep.baseline_sharpe > min_sharpe:
        lo, hi = 1.0, 64.0
        if run(strat, bars, costs.scaled(hi), ppy).metrics["sharpe"] > min_sharpe:
            rep.cost_death_bps = None  # survives even absurd costs
        else:
            for _ in range(12):
                mid = (lo + hi) / 2
                if run(strat, bars, costs.scaled(mid), ppy).metrics["sharpe"] > min_sharpe:
                    lo = mid
                else:
                    hi = mid
            per_turn_bps = (costs.commission_bps + costs.slippage_bps + costs.spread_bps)
            rep.cost_death_bps = round(lo * per_turn_bps, 2)


def _probe_worst_windows(rep: FragilityReport, strat: Strategy, bars: Bars,
                         costs: Costs, ppy: int, k: int, window: int,
                         min_sharpe: float) -> None:
    """Replay through the worst drawdown windows in the data, computed rather
    than cherry-picked."""
    for w in worst_windows(bars, k, window):
        start, end = int(w["start"]), int(w["end"])
        ctx = min(start, 250)
        sub = bars.slice(start - ctx, end)
        res = run(strat, sub, costs, ppy, warmup=ctx)
        s = res.metrics["sharpe"]
        dd = res.metrics["max_drawdown"]
        # Compare like with like: total return across this window, not an
        # annualised rate. A strategy is allowed to lose here; it is not allowed
        # to lose more than simply holding the asset did.
        total = 1.0
        for r in res.returns:
            total *= (1.0 + r)
        total -= 1.0
        survived = total >= float(w["return"])
        rep.probes.append(Probe(
            f"worst_window_{w['start_date']}",
            f"replay {w['start_date']}..{w['end_date']} (asset {float(w['return']):.1%}, "
            f"strategy {total:.1%})",
            survived, s, rep.baseline_sharpe,
            {"asset_return": w["return"], "strategy_return": total,
             "strategy_drawdown": dd,
             "start": w["start_date"], "end": w["end_date"]}))


def _probe_regimes(rep: FragilityReport, returns: Sequence[float], bars: Bars,
                   ppy: int, min_sharpe: float) -> None:
    """Which regimes bleed, and does the whole edge come from just one?"""
    labels = label_regimes(bars)[-len(returns):]
    breakdown = regime_breakdown(returns, labels, ppy)
    for label, m in breakdown.items():
        if m["sharpe"] < min_sharpe:
            rep.fatal_regimes.append(label)
        rep.probes.append(Probe(
            f"regime_{label}", f"performance inside {label} ({int(m['n'])} bars)",
            m["sharpe"] >= min_sharpe, m["sharpe"], rep.baseline_sharpe,
            {"share": m["share"], "total_return": m["total_return"]}))

    # Concentration: drop the single best regime and see what is left.
    if breakdown:
        best = max(breakdown, key=lambda k: breakdown[k]["total_return"])
        kept = [r for r, lab in zip(returns, labels) if lab != best]
        s = sharpe(kept, ppy) if len(kept) > 2 else 0.0
        rep.probes.append(Probe(
            "drop_best_regime",
            f"edge with the best regime ({best}) removed",
            s > min_sharpe, s, rep.baseline_sharpe, {"dropped": best}))


def _probe_lag(rep: FragilityReport, strat: Strategy, bars: Bars, costs: Costs,
               ppy: int, min_sharpe: float) -> None:
    """One extra bar of execution delay. Anything that dies here was living on
    fills it would never have got."""
    lagged = Strategy(strat.name + "_lag1",
                      lambda b, **kw: [0.0] + list(strat.weights(b))[:-1],
                      {}, "one-bar execution delay")
    s = run(lagged, bars, costs, ppy).metrics["sharpe"]
    rep.probes.append(Probe("execution_lag_1bar", "signal acted on one bar later",
                            s > min_sharpe, s, rep.baseline_sharpe, {}))


def _probe_params(rep: FragilityReport, strat: Strategy, bars: Bars, costs: Costs,
                  ppy: int, perturb: float, min_sharpe: float) -> None:
    """Nudge each numeric parameter. A real edge is a plateau; an overfit is a
    spike you happened to land on."""
    numeric = {k: v for k, v in strat.params.items() if isinstance(v, (int, float))
               and not isinstance(v, bool)}
    if not numeric:
        return
    sharpes = []
    for key, value in numeric.items():
        for direction in (1 - perturb, 1 + perturb):
            nv = type(value)(max(1, round(value * direction))) if isinstance(value, int) \
                else value * direction
            if nv == value:
                continue
            try:
                variant = get(strat.name, **{**strat.params, key: nv})
            except KeyError:
                variant = strat.with_params(**{key: nv})
            sharpes.append(run(variant, bars, costs, ppy).metrics["sharpe"])
    if not sharpes:
        return
    worst, spread = min(sharpes), stdev(sharpes) if len(sharpes) > 1 else 0.0
    # Survives if neighbours are also viable and the neighbourhood is not wild.
    survived = worst > min_sharpe and (
        spread <= max(abs(rep.baseline_sharpe), 0.1))
    rep.probes.append(Probe(
        "param_neighbourhood",
        f"+/-{perturb:.0%} on {', '.join(numeric)} ({len(sharpes)} neighbours)",
        survived, worst, rep.baseline_sharpe,
        {"worst_neighbour": worst, "spread": spread,
         "median_neighbour": percentile(sorted(sharpes), 0.5)}))


def _probe_null(rep: FragilityReport, strat: Strategy, base, bars: Bars,
                costs: Costs, ppy: int, samples: int) -> None:
    """Compare against random signals matched to the strategy's own exposure.

    If coin flips with the same time-in-market do this well this often, the
    strategy is not the reason for the returns.
    """
    if samples <= 0:
        return
    p_long = max(0.0, min(1.0, base.metrics.get("time_in_market", 0.5)))
    nulls = []
    for seed in range(samples):
        n = get("random_signal", seed=seed, p_long=p_long)
        nulls.append(run(n, bars, costs, ppy).metrics["sharpe"])
    beaten = sum(1 for s in nulls if s >= rep.baseline_sharpe)
    p = (beaten + 1) / (len(nulls) + 1)
    rep.null_p_value = p
    rep.probes.append(Probe(
        "vs_random_null", f"beat {samples} exposure-matched random signals",
        p < 0.05, rep.baseline_sharpe, percentile(sorted(nulls), 0.95),
        {"p_value": p, "null_p95": percentile(sorted(nulls), 0.95),
         "null_median": percentile(sorted(nulls), 0.5)}))
