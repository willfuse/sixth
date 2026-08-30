"""sixth -- the missing part of the quant research loop.

Research, code, backtest, live and post-mortem all work today. The sixth part,
folding what you learned back in so the next cycle starts smarter, is the one
that turns a trading bot into something that compounds. It is missing because
nothing keeps the results: a losing test writes a log line and the next strategy
an agent proposes has never heard of it.

This package is that store, plus enough honest machinery around it to produce
results worth keeping.

    from sixth import HypothesisGraph, Expectation, run_cycle, synthetic_bars

    g = HypothesisGraph("research.sqlite")
    h = g.add("A 20/100 MA cross beats buy-and-hold risk-adjusted.", tags=["trend"])
    g.preregister(h.id, Expectation.parse(["sharpe >= 0.8", "max_drawdown <= 0.25"]))
    result = run_cycle(g, h.id, "sma_cross", synthetic_bars(),
                       {"fast": [10, 20, 40], "slow": [60, 100, 200]})
    print(result.verdict)

Nothing here places a real order or holds a venue credential. See live.py.
"""

from .backtest import BacktestResult, Costs, run_backtest
from .breaker import FragilityReport, break_it
from .context import brief, payload
from .data import Bars, load_csv, synthetic_bars
from .graph import Hypothesis, HypothesisGraph
from .lessons import harvest, propose
from .live import KillSwitchTripped, PaperBroker, RiskGate, RiskLimits
from .loop import CycleResult, render_cycle, run_cycle
from .postmortem import PostMortem, analyse
from .prereg import Expectation, evaluate
from .regimes import label_regimes, regime_breakdown
from .strategy import Strategy, get_strategy, register
from .walkforward import seal_holdout, walk_forward

__version__ = "0.1.0"

__all__ = [
    "BacktestResult", "Costs", "run_backtest", "FragilityReport", "break_it",
    "brief", "payload", "Bars", "load_csv", "synthetic_bars", "Hypothesis",
    "HypothesisGraph", "harvest", "propose", "KillSwitchTripped", "PaperBroker",
    "RiskGate", "RiskLimits", "CycleResult", "render_cycle", "run_cycle",
    "PostMortem", "analyse", "Expectation", "evaluate", "label_regimes",
    "regime_breakdown", "Strategy", "get_strategy", "register", "seal_holdout",
    "walk_forward", "__version__",
]
