"""End-to-end: the whole argument of the package, exercised."""

import pytest

from sixth import context as ctx
from sixth.backtest import Costs
from sixth.breaker import break_it
from sixth.data import synthetic_bars
from sixth.graph import HypothesisGraph
from sixth.lessons import harvest, propose
from sixth.loop import run_cycle
from sixth.postmortem import analyse
from sixth.prereg import Expectation
from sixth.strategy import get


def seed_cycle(graph, bars, must=("sharpe >= 0.8",), strategy="sma_cross",
               grid=None, **kw):
    h = graph.add("A moving-average cross beats buy-and-hold risk-adjusted.",
                  tags=["trend"])
    graph.preregister(h.id, Expectation.parse(list(must), rationale="trends persist"))
    return h, run_cycle(graph, h.id, strategy, bars,
                        grid or {"fast": [10, 20], "slow": [60, 100]},
                        Costs(), n_folds=3, null_samples=25, **kw)


def test_cycle_refuses_to_run_without_a_sealed_expectation(graph, bars):
    h = graph.add("an unsealed idea")
    with pytest.raises(RuntimeError, match="preregistration"):
        run_cycle(graph, h.id, "sma_cross", bars, n_folds=3, null_samples=5)


def test_cycle_records_everything_it_learned(graph, bars):
    h, c = seed_cycle(graph, bars)
    assert c.verdict in ("confirmed", "refuted", "regime_conditional", "inconclusive")
    assert graph.get(h.id).status == c.verdict
    assert len(graph.experiments(h.id)) == 1
    assert graph.lessons(h.id), "a completed cycle must leave lessons behind"
    assert c.postmortem is not None and c.fragility is not None


def test_the_prereg_is_consumed_so_it_cannot_be_reused(graph, bars):
    h, _ = seed_cycle(graph, bars)
    assert graph.open_prereg(h.id) is None
    with pytest.raises(RuntimeError, match="preregistration"):
        run_cycle(graph, h.id, "sma_cross", bars, n_folds=3, null_samples=5)


def test_trial_count_accumulates_across_cycles(graph, bars):
    _, first = seed_cycle(graph, bars)
    assert first.n_trials == first.walkforward.n_trials
    _, second = seed_cycle(graph, bars)
    # The second cycle knows what the first one already spent.
    assert second.n_trials > second.walkforward.n_trials
    assert second.sealed.metrics["prior_trials"] == first.walkforward.n_trials


def test_deflated_sharpe_tightens_as_the_programme_spends_trials(graph, bars):
    """The whole point of keeping the graph: the bar rises with every search."""
    from sixth.stats import deflated_sharpe
    _, c = seed_cycle(graph, bars)
    r = c.sealed.returns
    naive = deflated_sharpe(r, 1)
    honest = deflated_sharpe(r, c.n_trials)
    assert honest <= naive


def test_sealed_holdout_is_not_used_for_fitting(graph, bars):
    """Fold windows must all sit inside the development half."""
    _, c = seed_cycle(graph, bars)
    dev_len = int(len(bars) * 0.8)
    for f in c.walkforward.folds:
        assert f.test[1] <= dev_len
        assert f.train[1] <= dev_len


def test_consensus_params_are_used_not_the_best_fold(graph, bars):
    from sixth.loop import _consensus_params
    _, c = seed_cycle(graph, bars)
    chosen = _consensus_params(c.walkforward)
    picks = [f.params for f in c.walkforward.folds]
    assert picks.count(chosen) >= len(picks) / len(set(map(repr, picks)))


def test_proposals_are_created_and_linked_when_asked(graph, bars):
    h, c = seed_cycle(graph, bars, create_proposals=True)
    if not c.proposals:
        pytest.skip("this run produced no follow-up proposals")
    created = [p for p in c.proposals if p["status"].startswith("created")]
    assert created
    frontier = [x.slug for x in graph.never_tried()]
    assert any(p["status"].split()[-1] in frontier for p in created)


def test_proposals_are_not_created_by_default(graph, bars):
    _, c = seed_cycle(graph, bars)
    assert all(not p["status"].startswith("created") for p in c.proposals)


def test_context_brief_names_what_already_failed(graph, bars):
    h, c = seed_cycle(graph, bars)
    brief = ctx.brief(graph)
    assert "Prior research state" in brief
    assert str(c.n_trials) in brief
    assert h.statement[:30] in brief


def test_context_payload_is_json_serialisable(graph, bars):
    import json
    seed_cycle(graph, bars)
    payload = json.loads(ctx.as_json(graph))
    assert payload["summary"]["experiments"] == 1
    assert "frontier" in payload and "refuted" in payload


def test_context_of_an_empty_graph_still_renders(graph):
    brief = ctx.brief(graph)
    assert "0 hypotheses" in brief


def test_experiment_stores_the_data_fingerprint(graph, bars):
    h, _ = seed_cycle(graph, bars)
    exp = graph.experiments(h.id)[0]
    assert exp.data_fingerprint == bars.fingerprint()
    assert exp.code_hash


def test_a_different_dataset_produces_a_different_fingerprint(bars):
    other = synthetic_bars(n=len(bars), seed=999)
    assert bars.fingerprint() != other.fingerprint()


def test_harvest_is_idempotent(graph, bars):
    from sixth.backtest import run as bt
    h = graph.add("idea")
    res = bt(get("sma_cross"), bars)
    pm = analyse(res, bars)
    fr = break_it(get("sma_cross"), bars, null_samples=10)
    harvest(graph, h.id, None, postmortem=pm, fragility=fr)
    first = len(graph.lessons(h.id))
    harvest(graph, h.id, None, postmortem=pm, fragility=fr)
    assert len(graph.lessons(h.id)) == first, "re-harvesting must not duplicate"
