"""The backtest engine.

Deliberately boring. The article is right that "the engine matters less than the
discipline around it" -- so this file does one thing carefully (execute weights
with realistic frictions and no lookahead) and leaves the discipline to
walkforward.py, breaker.py and graph.py.

Execution model
---------------
Weight for bar i is computed from data through bar i, and filled at bar i+1's
open. Costs are charged on the traded notional. Nothing is filled on the bar that
produced the signal, ever.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .data import Bars
from .stats import equity_curve, monte_carlo, summarize
from .strategy import Strategy


@dataclass
class Costs:
    """Frictions. Defaults are liquid-US-equity-ish; raise them for anything else."""

    commission_bps: float = 1.0     # per side, on traded notional
    slippage_bps: float = 2.0       # per side
    spread_bps: float = 0.0         # half-spread crossed, per side
    borrow_bps_annual: float = 50.0  # cost of holding short, annualised

    def per_turn(self) -> float:
        return (self.commission_bps + self.slippage_bps + self.spread_bps) / 10_000.0

    def scaled(self, factor: float) -> "Costs":
        return Costs(self.commission_bps * factor, self.slippage_bps * factor,
                     self.spread_bps * factor, self.borrow_bps_annual * factor)


@dataclass
class BacktestResult:
    strategy: str
    params: Dict[str, Any]
    symbol: str
    dates: List[str]
    returns: List[float] = field(default_factory=list)       # net, per bar
    gross_returns: List[float] = field(default_factory=list)
    weights: List[float] = field(default_factory=list)       # as held
    costs_paid: List[float] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    data_fingerprint: str = ""
    code_hash: str = ""

    @property
    def equity(self) -> List[float]:
        return equity_curve(self.returns)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy, "params": self.params, "symbol": self.symbol,
            "n_bars": len(self.returns), "start": self.dates[0] if self.dates else None,
            "end": self.dates[-1] if self.dates else None, "metrics": self.metrics,
            "data_fingerprint": self.data_fingerprint, "code_hash": self.code_hash,
        }


def run(strategy: Strategy, bars: Bars, costs: Optional[Costs] = None,
        periods_per_year: int = 252, max_leverage: float = 1.0,
        monte_carlo_paths: int = 0, warmup: int = 0) -> BacktestResult:
    """Execute `strategy` over `bars` and return net-of-cost results."""
    costs = costs or Costs()
    raw = strategy.weights(bars)
    n = len(bars)
    closes, opens = bars.close, bars.open

    held = 0.0
    net_rets: List[float] = []
    gross_rets: List[float] = []
    held_series: List[float] = []
    cost_series: List[float] = []
    turn_cost = costs.per_turn()
    borrow_per_bar = costs.borrow_bps_annual / 10_000.0 / periods_per_year

    for i in range(n):
        # Position entering bar i was decided at bar i-1's close.
        target = 0.0 if i == 0 else max(-max_leverage,
                                        min(max_leverage, raw[i - 1] * max_leverage))
        traded = abs(target - held)

        # Cost of getting to `target` is paid against bar i's open.
        cost = traded * turn_cost
        if target < 0:
            cost += abs(target) * borrow_per_bar
        held = target

        if i == 0:
            gross = 0.0
        else:
            # Held from bar i-1 close into bar i close, entering at bar i open:
            # the gap is taken at the traded weight, the intraday leg at `held`.
            prev_close = closes[i - 1]
            gap = (opens[i] - prev_close) / prev_close if prev_close else 0.0
            intraday = (closes[i] - opens[i]) / opens[i] if opens[i] else 0.0
            gross = held * (gap + intraday)

        gross_rets.append(gross)
        cost_series.append(cost)
        net_rets.append(gross - cost)
        held_series.append(held)

    if warmup:
        sl = slice(warmup, None)
        net_rets, gross_rets = net_rets[sl], gross_rets[sl]
        held_series, cost_series = held_series[sl], cost_series[sl]
        dates = bars.dates[sl]
    else:
        dates = list(bars.dates)

    metrics = summarize(net_rets, periods_per_year)
    metrics.update({
        "gross_sharpe": summarize(gross_rets, periods_per_year)["sharpe"],
        "total_costs": sum(cost_series),
        "turnover_annual": _turnover(held_series) / max(len(held_series), 1) * periods_per_year,
        "time_in_market": sum(1 for w in held_series if w != 0) / max(len(held_series), 1),
        "avg_exposure": sum(held_series) / max(len(held_series), 1),
    })
    if monte_carlo_paths:
        metrics.update(monte_carlo(net_rets, monte_carlo_paths,
                                   periods_per_year=periods_per_year))

    return BacktestResult(
        strategy=strategy.name, params=dict(strategy.params), symbol=bars.symbol,
        dates=dates, returns=net_rets, gross_returns=gross_rets,
        weights=held_series, costs_paid=cost_series, metrics=metrics,
        data_fingerprint=bars.fingerprint(), code_hash=strategy.code_hash(),
    )


def _turnover(weights: Sequence[float]) -> float:
    return sum(abs(weights[i] - weights[i - 1]) for i in range(1, len(weights)))


def benchmark(bars: Bars, costs: Optional[Costs] = None,
              periods_per_year: int = 252) -> BacktestResult:
    """Buy-and-hold on the same bars, same frictions. Every result should be read
    against this, not against zero."""
    from .strategy import get
    return run(get("buy_hold"), bars, costs, periods_per_year)


#: Public alias. `run` reads well inside the package; from outside,
#: `sixth.run_backtest` avoids shadowing the `sixth.backtest` module itself.
run_backtest = run
