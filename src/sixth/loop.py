"""One turn of the loop, wired end to end.

    prereg -> seal holdout -> walk-forward -> break -> sealed OOS -> post-mortem
           -> verdict -> record -> harvest lessons -> propose next

Two things this refuses to do, both deliberate:

  * Run a hypothesis with no sealed expectation. Without one there is no verdict,
    only a number you get to interpret afterwards -- which is the failure mode the
    whole package exists to prevent.
  * Report a Deflated Sharpe against a trial count of 1 when the graph knows you
    have spent hundreds. `n_trials` is read from the graph, not from this run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .backtest import BacktestResult, Costs, run
from .breaker import FragilityReport, break_it
from .data import Bars
from .graph import HypothesisGraph
from .lessons import harvest, propose
from .postmortem import PostMortem, analyse
from .prereg import Expectation, evaluate
from .stats import deflated_sharpe
from .strategy import get
from .walkforward import WalkForwardResult, seal_holdout, walk_forward


@dataclass
class CycleResult:
    hypothesis: str
    verdict: str
    experiment_id: int
    walkforward: Optional[WalkForwardResult] = None
    sealed: Optional[BacktestResult] = None
    fragility: Optional[FragilityReport] = None
    postmortem: Optional[PostMortem] = None
    lessons: List[Dict[str, str]] = field(default_factory=list)
    proposals: List[Dict[str, str]] = field(default_factory=list)
    n_trials: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis": self.hypothesis, "verdict": self.verdict,
            "experiment_id": self.experiment_id, "n_trials": self.n_trials,
            "walkforward": self.walkforward.to_dict() if self.walkforward else None,
            "sealed": self.sealed.to_dict() if self.sealed else None,
            "fragility": self.fragility.to_dict() if self.fragility else None,
            "postmortem": self.postmortem.to_dict() if self.postmortem else None,
            "lessons": self.lessons, "proposals": self.proposals,
        }


def run_cycle(graph: HypothesisGraph, ref: Any, strategy_name: str, bars: Bars,
              grid: Optional[Dict[str, Sequence[Any]]] = None,
              costs: Optional[Costs] = None, holdout: float = 0.2,
              n_folds: int = 5, embargo: int = 10, min_train: int = 250,
              periods_per_year: int = 252, null_samples: int = 200,
              create_proposals: bool = False,
              count_prior_trials: bool = True) -> CycleResult:
    """Run the full cycle for one hypothesis and write everything to the graph."""
    h = graph.get(ref)
    opened = graph.open_prereg(h.id)
    if opened is None:
        raise RuntimeError(
            f"{h.slug} has no open preregistration. Seal an expectation first "
            f"(`sixth prereg {h.slug} --must \"sharpe >= 0.8\"`) -- a test with no "
            f"stated expectation cannot produce a verdict, only a number.")
    prereg_id, expectation = opened
    costs = costs or Costs()

    # 1. Hide the final slice before anything looks at the data.
    dev, sealed_bars = seal_holdout(bars, holdout)

    # 2. Walk forward on the development half only.
    wf = walk_forward(strategy_name, dev, grid, costs, n_folds, embargo,
                      min_train, periods_per_year)

    # 3. Adversarial pass, also on the development half.
    chosen = _consensus_params(wf)
    strat = get(strategy_name, **chosen)
    fragility = break_it(strat, dev, costs, periods_per_year,
                         null_samples=null_samples)

    # 4. One look at the sealed window, with the parameters now frozen.
    sealed_res = run(strat, sealed_bars, costs, periods_per_year)

    # 5. Honest trial count: this run's search plus everything the graph
    #    already watched being spent on this programme.
    prior = graph.trial_count() if count_prior_trials else 0
    n_trials = wf.n_trials + prior
    sealed_res.metrics["dsr"] = deflated_sharpe(
        sealed_res.returns, max(n_trials, 1), periods_per_year=periods_per_year)
    sealed_res.metrics["n_trials"] = float(n_trials)
    sealed_res.metrics["prior_trials"] = float(prior)
    sealed_res.metrics["oos_sharpe_walkforward"] = wf.metrics.get("sharpe", 0.0)
    sealed_res.metrics["sharpe_decay"] = wf.sharpe_decay
    sealed_res.metrics["param_stability"] = wf.metrics.get("param_stability", 0.0)
    sealed_res.metrics["robustness"] = fragility.robustness
    sealed_res.metrics["null_p_value"] = fragility.null_p_value

    # 6. Autopsy and verdict, judged against the sealed expectation.
    pm = analyse(sealed_res, sealed_bars, expectation, periods_per_year)
    verdict, checks = evaluate(expectation, sealed_res.metrics, pm.regimes)
    pm.verdict = verdict

    # 7. Append the result. Nothing above can be revised after this point.
    exp_id = graph.record(
        h.id, verdict, sealed_res.metrics,
        config={"strategy": strategy_name, "params": chosen, "grid": dict(grid or {}),
                "costs": vars(costs), "holdout": holdout, "n_folds": n_folds,
                "embargo": embargo, "symbol": bars.symbol},
        regimes=pm.regimes,
        checks=[vars(c) for c in checks], prereg_id=prereg_id,
        data_fingerprint=bars.fingerprint(), code_hash=strat.code_hash(),
        n_trials=wf.n_trials,
        notes=f"walk-forward {n_folds} folds, holdout {holdout:.0%}")

    # 8. Fold the lesson back in. This is the part that makes the next cycle
    #    different from this one.
    written = harvest(graph, h.id, exp_id, postmortem=pm, fragility=fragility,
                      walkforward=wf)
    proposals = propose(graph, h.id, postmortem=pm, fragility=fragility,
                        create=create_proposals)

    return CycleResult(h.slug, verdict, exp_id, wf, sealed_res, fragility, pm,
                       written, proposals, n_trials)


def _consensus_params(wf: WalkForwardResult) -> Dict[str, Any]:
    """The parameter set most folds agreed on.

    Not the best-performing one. Picking the best across folds is peeking at the
    test windows, which is the exact leak walk-forward exists to prevent.
    """
    if not wf.folds:
        return {}
    counts: Dict[str, int] = {}
    lookup: Dict[str, Dict[str, Any]] = {}
    for f in wf.folds:
        key = repr(sorted(f.params.items()))
        counts[key] = counts.get(key, 0) + 1
        lookup[key] = f.params
    winner = max(counts, key=lambda k: (counts[k], k))
    return lookup[winner]


def render_cycle(c: CycleResult) -> str:
    """Terminal summary of one cycle."""
    L: List[str] = []
    L.append("=" * 68)
    L.append(f"CYCLE COMPLETE: {c.hypothesis}")
    L.append(f"VERDICT: {c.verdict.upper()}   (experiment #{c.experiment_id})")
    L.append("=" * 68)

    if c.walkforward:
        m = c.walkforward.metrics
        L.append("")
        L.append("WALK-FORWARD (development data)")
        L.append(f"  folds                {len(c.walkforward.folds)}")
        L.append(f"  out-of-sample Sharpe {m.get('sharpe', 0):.3f}")
        L.append(f"  Sharpe decay IS->OOS {c.walkforward.sharpe_decay:.3f}")
        L.append(f"  parameter stability  {m.get('param_stability', 0):.0%}")
        L.append(f"  folds profitable     {m.get('fold_win_rate', 0):.0%}")

    if c.fragility:
        L.append("")
        L.append("BREAKER (adversarial)")
        L.append(f"  probes survived      {c.fragility.robustness:.0%} "
                 f"({len(c.fragility.probes) - len(c.fragility.failures)}"
                 f"/{len(c.fragility.probes)})")
        if c.fragility.cost_death_bps is not None:
            L.append(f"  edge dies at         {c.fragility.cost_death_bps:.0f} bps round-turn")
        L.append(f"  vs random null       p = {c.fragility.null_p_value:.3f}")
        for p in c.fragility.failures[:6]:
            L.append(f"    KILLED  {p.name}: {p.description}")

    if c.sealed:
        m = c.sealed.metrics
        L.append("")
        L.append("SEALED HOLDOUT (touched once, just now)")
        L.append(f"  Sharpe               {m.get('sharpe', 0):.3f}")
        L.append(f"  max drawdown         {m.get('max_drawdown', 0):.1%}")
        L.append(f"  Deflated Sharpe      {m.get('dsr', 0):.3f} "
                 f"over {int(m.get('n_trials', 1))} trials "
                 f"({int(m.get('prior_trials', 0))} already spent before this run)")

    if c.postmortem:
        L.append("")
        L.append("FINDINGS")
        for f in c.postmortem.findings:
            L.append(f"  - {f}")

    if c.lessons:
        L.append("")
        L.append(f"WRITTEN TO GRAPH ({len(c.lessons)} lessons)")
        for l in c.lessons:
            tag = f"[{l['kind']}]"
            L.append(f"  {tag:<16} {l['text']}")

    if c.proposals:
        L.append("")
        L.append("NEXT HYPOTHESES IMPLIED BY THIS FAILURE")
        for p in c.proposals:
            L.append(f"  - {p['statement']}")
            L.append(f"      because: {p['reason']}  [{p['status']}]")
    return "\n".join(L)
