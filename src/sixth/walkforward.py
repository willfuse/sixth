"""Walk-forward validation and the sealed out-of-sample window.

Two rules from the article, made mechanical:

  "Sealed out-of-sample windows the developer never touches."
  "Deflated Sharpe."

`seal_holdout` physically removes the final slice of the data before you are
allowed to see anything, and `walk_forward` counts every parameter combination it
evaluates -- that count is what makes a Deflated Sharpe honest instead of
decorative.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .backtest import BacktestResult, Costs, run
from .data import Bars
from .stats import deflated_sharpe, sharpe, summarize
from .strategy import Strategy, get


@dataclass
class Fold:
    index: int
    train: Tuple[int, int]
    test: Tuple[int, int]
    params: Dict[str, Any] = field(default_factory=dict)
    train_sharpe: float = 0.0
    test_sharpe: float = 0.0
    n_candidates: int = 0


@dataclass
class WalkForwardResult:
    strategy: str
    folds: List[Fold]
    oos_returns: List[float]
    oos_dates: List[str]
    metrics: Dict[str, float] = field(default_factory=dict)
    n_trials: int = 0

    @property
    def sharpe_decay(self) -> float:
        """In-sample Sharpe minus out-of-sample Sharpe, averaged over folds.

        Large positive decay is the signature of a fit to noise. It is the number
        to write into the graph, because it predicts what the next variant of the
        same idea will do.
        """
        if not self.folds:
            return 0.0
        return sum(f.train_sharpe - f.test_sharpe for f in self.folds) / len(self.folds)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy, "n_folds": len(self.folds),
            "n_trials": self.n_trials, "sharpe_decay": self.sharpe_decay,
            "metrics": self.metrics,
            "folds": [{"index": f.index, "train": f.train, "test": f.test,
                       "params": f.params, "train_sharpe": f.train_sharpe,
                       "test_sharpe": f.test_sharpe, "n_candidates": f.n_candidates}
                      for f in self.folds],
        }


def seal_holdout(bars: Bars, frac: float = 0.2) -> Tuple[Bars, Bars]:
    """Split into (development, sealed). Touch the sealed half once, at the end.

    Returning it as a separate object is the whole mechanism: you cannot
    accidentally include what you were never handed.
    """
    if not 0.0 < frac < 1.0:
        raise ValueError("frac must be in (0, 1)")
    cut = int(len(bars) * (1.0 - frac))
    return bars.slice(0, cut), bars.slice(cut)


def purged_splits(n: int, n_folds: int = 5, embargo: int = 10,
                  min_train: int = 250, anchored: bool = True
                  ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Expanding-window splits with an embargo gap between train and test.

    The embargo drops the bars straddling the boundary. Without it, a strategy
    using a 100-bar lookback trains on features built from the first bars of its
    own test set.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    # The first test window starts after a full training window *plus* the
    # embargo, so no fold is ever trained on less than min_train bars.
    first_test = min_train + embargo
    usable = n - first_test
    if usable < n_folds * 2:
        raise ValueError(
            f"{n} bars is not enough for {n_folds} folds with min_train={min_train} "
            f"and embargo={embargo}")
    test_size = usable // n_folds
    splits = []
    for k in range(n_folds):
        test_start = first_test + k * test_size
        test_end = test_start + test_size if k < n_folds - 1 else n
        train_end = test_start - embargo
        train_start = 0 if anchored else max(0, train_end - min_train)
        splits.append(((train_start, train_end), (test_start, test_end)))
    return splits


def expand_grid(grid: Dict[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def walk_forward(strategy_name: str, bars: Bars,
                 grid: Optional[Dict[str, Sequence[Any]]] = None,
                 costs: Optional[Costs] = None, n_folds: int = 5,
                 embargo: int = 10, min_train: int = 250,
                 periods_per_year: int = 252,
                 objective: str = "sharpe") -> WalkForwardResult:
    """Fit parameters on each training window, score on the next unseen window.

    Every candidate evaluated anywhere counts toward `n_trials`. If you sweep 20
    parameter sets over 5 folds you have run 100 trials, and the Deflated Sharpe
    below is computed against that -- not against 1.
    """
    costs = costs or Costs()
    candidates = expand_grid(grid or {})
    splits = purged_splits(len(bars), n_folds, embargo, min_train)

    folds: List[Fold] = []
    oos_returns: List[float] = []
    oos_dates: List[str] = []
    trials = 0

    for i, ((tr0, tr1), (te0, te1)) in enumerate(splits):
        train_bars = bars.slice(tr0, tr1)
        best, best_score = None, float("-inf")
        for params in candidates:
            trials += 1
            res = run(get(strategy_name, **params), train_bars, costs, periods_per_year)
            score = res.metrics.get(objective, float("-inf"))
            if score > best_score:
                best, best_score = params, score
        best = best or {}

        # Test window carries min_train bars of context so indicators are warm,
        # then only the post-warmup returns are kept.
        ctx = min(te0, min_train)
        test_bars = bars.slice(te0 - ctx, te1)
        test_res = run(get(strategy_name, **best), test_bars, costs,
                       periods_per_year, warmup=ctx)

        folds.append(Fold(i, (tr0, tr1), (te0, te1), best, best_score,
                          test_res.metrics.get("sharpe", 0.0), len(candidates)))
        oos_returns.extend(test_res.returns)
        oos_dates.extend(test_res.dates)

    metrics = summarize(oos_returns, periods_per_year)
    metrics["dsr"] = deflated_sharpe(oos_returns, max(trials, 1),
                                     periods_per_year=periods_per_year)
    metrics["n_trials"] = float(trials)
    metrics["sharpe_decay"] = (
        sum(f.train_sharpe - f.test_sharpe for f in folds) / len(folds) if folds else 0.0)
    metrics["fold_sharpe_min"] = min((f.test_sharpe for f in folds), default=0.0)
    metrics["fold_win_rate"] = (
        sum(1 for f in folds if f.test_sharpe > 0) / len(folds) if folds else 0.0)
    metrics["param_stability"] = _param_stability(folds)

    return WalkForwardResult(strategy_name, folds, oos_returns, oos_dates,
                             metrics, trials)


def _param_stability(folds: Sequence[Fold]) -> float:
    """Share of folds that chose the single most common parameter set.

    A strategy whose optimal parameters jump every fold does not have an edge; it
    has a curve-fit that keeps needing a new curve.
    """
    if not folds:
        return 0.0
    keys = [repr(sorted(f.params.items())) for f in folds]
    return max(keys.count(k) for k in set(keys)) / len(keys)
