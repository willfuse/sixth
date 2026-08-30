"""Fine-tune: fold the lesson back in.

This is the step the article says nobody has built. Not because it is hard --
because nothing was keeping the results. Given a graph, it is mechanical: turn
each post-mortem finding and each breaker kill into a typed, regime-tagged row,
attach it to the hypothesis that produced it, and propose the next hypotheses the
evidence actually implies.

The output of a losing test stops being a log line and becomes an input.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .breaker import FragilityReport
from .graph import HypothesisGraph
from .postmortem import PostMortem
from .walkforward import WalkForwardResult

# Lesson kinds. Kept small on purpose -- a taxonomy nobody can remember does not
# get used, and unused structure is worse than none.
FRAGILITY = "fragility"       # it breaks under condition X
REGIME = "regime"             # it works / fails specifically in regime X
EXECUTION = "execution"       # costs, turnover, fills
OVERFIT = "overfit"           # the search, not the market, produced the result
CONCENTRATION = "concentration"  # the track record rests on very few bars
NULL = "null"                 # indistinguishable from chance


def harvest(graph: HypothesisGraph, ref: Any, experiment_id: Optional[int] = None,
            postmortem: Optional[PostMortem] = None,
            fragility: Optional[FragilityReport] = None,
            walkforward: Optional[WalkForwardResult] = None) -> List[Dict[str, str]]:
    """Extract lessons from a completed test and write them into the graph.

    Returns what was written, so a caller can show it without re-querying.
    """
    written: List[Dict[str, str]] = []

    def put(kind: str, text: str, regime: str = "",
            evidence: Optional[Dict[str, Any]] = None) -> None:
        graph.add_lesson(kind, text, ref=ref, experiment_id=experiment_id,
                         regime=regime, evidence=evidence or {})
        written.append({"kind": kind, "regime": regime, "text": text})

    if postmortem is not None:
        _from_postmortem(postmortem, put)
    if fragility is not None:
        _from_fragility(fragility, put)
    if walkforward is not None:
        _from_walkforward(walkforward, put)
    return written


def _from_postmortem(pm: PostMortem, put) -> None:
    for regime, r in pm.regimes.items():
        if r["sharpe"] < -0.3 and r["n"] >= 30:
            put(REGIME,
                f"Loses money in {regime}: Sharpe {r['sharpe']:.2f} over {int(r['n'])} bars "
                f"({r['total_return']:.1%} total). Do not run this unfiltered in {regime}.",
                regime=regime, evidence=r)
        elif r["sharpe"] > 0.8 and r["n"] >= 30:
            put(REGIME,
                f"Works in {regime}: Sharpe {r['sharpe']:.2f} over {int(r['n'])} bars. "
                f"The edge is conditional on this regime, not general.",
                regime=regime, evidence=r)

    turnover = pm.metrics.get("turnover_annual", 0)
    if pm.cost_drag == float("inf"):
        put(EXECUTION,
            f"Gross returns were negative before any costs, at {turnover:.1f}x annual "
            f"turnover. Cutting costs cannot rescue this; the signal itself is wrong.",
            evidence={"cost_drag": None, "turnover": turnover})
    elif pm.cost_drag > 1.0:
        put(EXECUTION,
            f"Frictions consumed {pm.cost_drag:.0%} of the gross edge at "
            f"{turnover:.1f}x annual turnover. Any variant of this idea must cut "
            f"turnover before it can work.",
            evidence={"cost_drag": pm.cost_drag, "turnover": turnover})

    if pm.concentration > 0.6:
        put(CONCENTRATION,
            f"{pm.concentration:.0%} of gains came from the best 5% of bars. Treat the reported "
            f"Sharpe as unreliable and require a longer sample before believing it.",
            evidence={"concentration": pm.concentration})

    if pm.episodes and not pm.episodes[0].recovered:
        e = pm.episodes[0]
        put(FRAGILITY,
            f"Ended the sample {e.depth:.0%} underwater, in drawdown since {e.start_date} "
            f"({e.regime}). The last observed state is a losing one.",
            regime=e.regime, evidence={"depth": e.depth, "since": e.start_date})


def _from_fragility(fr: FragilityReport, put) -> None:
    if fr.cost_death_bps is not None:
        put(EXECUTION,
            f"Edge disappears at roughly {fr.cost_death_bps:.0f} bps round-turn cost. "
            f"Only viable where execution is cheaper than that.",
            evidence={"cost_death_bps": fr.cost_death_bps})

    for p in fr.failures:
        if p.name.startswith("worst_window"):
            put(FRAGILITY,
                f"Underperformed simply holding the asset through {p.detail.get('start')}.."
                f"{p.detail.get('end')} ({p.detail.get('strategy_return', 0):.1%} vs "
                f"{p.detail.get('asset_return', 0):.1%}).",
                evidence=p.detail)
        elif p.name == "execution_lag_1bar":
            put(FRAGILITY,
                f"Dies with one bar of execution delay (Sharpe {p.metric:.2f}). The result "
                f"depends on fills you will not get.", evidence=p.detail)
        elif p.name == "param_neighbourhood":
            put(OVERFIT,
                f"Parameter neighbours fail: worst nearby setting scores {p.metric:.2f} against "
                f"{p.baseline:.2f} here. This is a spike in the parameter surface, not a plateau.",
                evidence=p.detail)
        elif p.name == "drop_best_regime":
            put(CONCENTRATION,
                f"Removing the single best regime ({p.detail.get('dropped')}) leaves Sharpe "
                f"{p.metric:.2f}. The edge is one regime wearing a strategy's clothes.",
                regime=str(p.detail.get("dropped", "")), evidence=p.detail)
        elif p.name == "vs_random_null":
            put(NULL,
                f"Does not beat exposure-matched random signals (p = "
                f"{p.detail.get('p_value', 1):.3f}). Indistinguishable from chance at this "
                f"sample size.", evidence=p.detail)
        elif p.name.startswith("costs_x"):
            put(EXECUTION,
                f"Fails at {p.detail.get('multiplier')}x baseline costs (Sharpe {p.metric:.2f}).",
                evidence=p.detail)


def _from_walkforward(wf: WalkForwardResult, put) -> None:
    m = wf.metrics
    if m.get("dsr", 1.0) < 0.5 and m.get("n_trials", 1) > 1:
        put(OVERFIT,
            f"Walk-forward Deflated Sharpe {m['dsr']:.2f} after {int(m['n_trials'])} trials "
            f"in this search: the out-of-sample edge is within what searching this hard "
            f"would produce from noise.",
            evidence={"dsr": m["dsr"], "n_trials": m["n_trials"]})

    if wf.sharpe_decay > 0.5:
        put(OVERFIT,
            f"Sharpe decays {wf.sharpe_decay:.2f} from in-sample to out-of-sample across "
            f"{len(wf.folds)} folds. Parameters are fitting the training window.",
            evidence={"decay": wf.sharpe_decay})

    if m.get("param_stability", 1.0) < 0.5:
        put(OVERFIT,
            f"Optimal parameters changed in most folds (stability "
            f"{m['param_stability']:.0%}). There is no stable setting to deploy.",
            evidence={"param_stability": m["param_stability"]})

    if m.get("fold_win_rate", 1.0) < 0.5:
        put(FRAGILITY,
            f"Profitable in only {m['fold_win_rate']:.0%} of walk-forward folds; worst fold "
            f"Sharpe {m.get('fold_sharpe_min', 0):.2f}.",
            evidence={"fold_win_rate": m["fold_win_rate"]})


# --------------------------------------------------------------------------
# proposing the next hypotheses
# --------------------------------------------------------------------------
def propose(graph: HypothesisGraph, ref: Any,
            postmortem: Optional[PostMortem] = None,
            fragility: Optional[FragilityReport] = None,
            create: bool = False) -> List[Dict[str, str]]:
    """Turn the failure into the next thing to try.

    Every proposal is linked back to the hypothesis it came from, so the graph
    records not just what failed but what the failure suggested -- and six months
    later you can see whether you ever followed up.
    """
    parent = graph.get(ref)
    out: List[Dict[str, str]] = []

    def add(statement: str, reason: str, kind: str = "refines") -> None:
        dup = graph.similar(statement, k=1)
        if dup and dup[0][1] > 0.92:
            out.append({"statement": statement, "reason": reason,
                        "status": f"skipped, near-duplicate of {dup[0][0].slug}"})
            return
        status = "proposed"
        if create:
            h = graph.add(statement, tags=parent.tags + ["proposed"],
                          parent=parent.slug, edge_kind=kind,
                          spec={"from_experiment_on": parent.slug, "reason": reason})
            status = f"created as {h.slug}"
        out.append({"statement": statement, "reason": reason, "status": status})

    if postmortem:
        bad = [r for r, v in postmortem.regimes.items() if v["sharpe"] < -0.3]
        good = [r for r, v in postmortem.regimes.items() if v["sharpe"] > 0.8]
        if bad:
            add(f"{parent.statement} -- but flat during {', '.join(sorted(bad))}",
                f"the base idea bled in {len(bad)} regime(s); a filter is the cheapest fix")
        if good and len(good) < len(postmortem.regimes):
            add(f"{parent.statement} -- traded only during {', '.join(sorted(good))}",
                "the edge appears regime-conditional; test it as such rather than generally")
        if postmortem.cost_drag == float("inf"):
            pass  # negative gross: turnover is not the problem, so do not propose that fix
        elif postmortem.cost_drag > 1.0:
            add(f"{parent.statement} -- with a holding-period floor to cut turnover",
                f"costs took {postmortem.cost_drag:.0%} of the gross edge")

    if fragility:
        if any(p.name == "execution_lag_1bar" and not p.survived for p in fragility.probes):
            add(f"{parent.statement} -- evaluated with a one-bar execution delay built in",
                "the result did not survive realistic fill timing")
        if any(p.name == "param_neighbourhood" and not p.survived for p in fragility.probes):
            add(f"{parent.statement} -- with parameters averaged over a neighbourhood "
                f"instead of a single optimum",
                "the parameter surface is a spike, so ensembling across it is the honest version")
    return out
