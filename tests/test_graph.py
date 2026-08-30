import sqlite3

import pytest

from sixth.graph import HypothesisGraph, slugify
from sixth.prereg import Expectation


def test_add_and_fetch(graph):
    h = graph.add("Momentum works in equities", tags=["trend", "equity"])
    assert h.status == "untested"
    assert graph.get(h.slug).id == h.id
    assert graph.get(h.id).slug == h.slug
    assert graph.get(h.id).tags == ["trend", "equity"]


def test_slugs_are_unique(graph):
    a = graph.add("Momentum works")
    b = graph.add("Momentum works")
    assert a.slug != b.slug


def test_missing_hypothesis_raises(graph):
    with pytest.raises(KeyError):
        graph.get("does-not-exist")
    assert graph.exists("does-not-exist") is False


def test_never_tried_lists_only_untested(graph):
    a = graph.add("untested idea")
    b = graph.add("tested idea")
    graph.record(b.id, "refuted", {"sharpe": -0.5})
    slugs = [h.slug for h in graph.never_tried()]
    assert a.slug in slugs
    assert b.slug not in slugs


def test_retired_is_excluded_from_frontier(graph):
    h = graph.add("abandoned idea")
    graph.set_status(h.id, "retired")
    assert h.slug not in [x.slug for x in graph.never_tried()]


def test_experiments_are_append_only(graph):
    h = graph.add("idea")
    graph.record(h.id, "refuted", {"sharpe": -0.2})
    with pytest.raises(sqlite3.IntegrityError):
        graph.conn.execute("UPDATE experiments SET verdict='confirmed'")
    with pytest.raises(sqlite3.IntegrityError):
        graph.conn.execute("DELETE FROM experiments")


def test_preregs_are_sealed(graph):
    h = graph.add("idea")
    graph.preregister(h.id, Expectation.parse(["sharpe >= 1.0"]))
    with pytest.raises(sqlite3.IntegrityError):
        graph.conn.execute("UPDATE preregs SET expectation_json='{}'")
    with pytest.raises(sqlite3.IntegrityError):
        graph.conn.execute("DELETE FROM preregs")


def test_seal_detects_tampering(graph):
    """The trigger blocks the honest path; the hash catches someone who goes
    around it."""
    h = graph.add("idea")
    graph.preregister(h.id, Expectation.parse(["sharpe >= 1.0"]))
    assert graph.verify_seals() == []
    graph.conn.execute("DROP TRIGGER preregs_no_update")
    graph.conn.execute(
        "UPDATE preregs SET expectation_json=?",
        (Expectation.parse(["sharpe >= 0.01"]).to_json(),))
    assert graph.verify_seals() != []
    assert graph.summary()["seal_failures"] == 1


def test_open_prereg_is_consumed_by_a_recorded_run(graph):
    h = graph.add("idea")
    pid = graph.preregister(h.id, Expectation.parse(["sharpe >= 1.0"]))
    assert graph.open_prereg(h.id)[0] == pid
    graph.record(h.id, "refuted", {"sharpe": 0.0}, prereg_id=pid)
    assert graph.open_prereg(h.id) is None


def test_trial_count_sums_and_filters_by_tag(graph):
    a = graph.add("trend idea", tags=["trend"])
    b = graph.add("flow idea", tags=["flow"])
    graph.record(a.id, "refuted", {}, n_trials=40)
    graph.record(b.id, "refuted", {}, n_trials=15)
    assert graph.trial_count() == 55
    assert graph.trial_count(tag="trend") == 40
    assert graph.trial_count(tag="flow") == 15


def test_refuted_in_finds_regime_losses_even_for_winners(graph):
    h = graph.add("mostly good idea")
    graph.record(h.id, "confirmed", {"sharpe": 1.2},
                 regimes={"up/calm": {"sharpe": 2.0}, "down/stressed": {"sharpe": -1.1}})
    assert [x.slug for x, _ in graph.refuted_in("down/stressed")] == [h.slug]
    assert graph.refuted_in("up/calm") == []


def test_confirmed_only_in_reports_working_regimes(graph):
    h = graph.add("conditional idea")
    graph.record(h.id, "regime_conditional", {"sharpe": 0.1},
                 regimes={"up/calm": {"sharpe": 1.5}, "down/normal": {"sharpe": -0.8}})
    got = graph.confirmed_only_in()
    assert len(got) == 1
    assert got[0][1] == ["up/calm"]


def test_similar_ranks_related_statements_first(graph):
    graph.add("A moving average crossover captures trend persistence", tags=["trend"])
    graph.add("Options order flow imbalance predicts overnight gaps", tags=["flow"])
    ranked = graph.similar("simple moving average trend following crossover")
    assert ranked
    assert "moving-average" in ranked[0][0].slug


def test_similar_works_on_a_two_row_graph(graph):
    """Smoothed IDF: a term in every document must still carry weight."""
    graph.add("trend following works")
    graph.add("trend following fails")
    assert graph.similar("trend following") != []


def test_already_tested_matches_on_code_hash(graph):
    h = graph.add("idea")
    graph.record(h.id, "refuted", {}, code_hash="abc123", data_fingerprint="data1")
    assert graph.already_tested("abc123")
    assert graph.already_tested("abc123", "data1")
    assert not graph.already_tested("abc123", "otherdata")
    assert not graph.already_tested("zzz")


def test_ancestry_walks_back_to_the_root(graph):
    root = graph.add("base idea")
    mid = graph.add("refined idea", parent=root.slug, edge_kind="refines")
    leaf = graph.add("further refined", parent=mid.slug, edge_kind="refines")
    assert [h.slug for h in graph.ancestry(leaf.slug)] == [mid.slug, root.slug]


def test_edges_are_deduplicated(graph):
    a, b = graph.add("a"), graph.add("b")
    graph.link(a.slug, b.slug, "refines")
    graph.link(a.slug, b.slug, "refines")
    assert graph.summary()["edges"] == 1


def test_invalid_edge_kind_and_status_are_rejected(graph):
    a, b = graph.add("a"), graph.add("b")
    with pytest.raises(ValueError):
        graph.link(a.slug, b.slug, "invented_kind")
    with pytest.raises(ValueError):
        graph.set_status(a.id, "vibes")


def test_lessons_are_deduplicated(graph):
    h = graph.add("idea")
    for _ in range(3):
        graph.add_lesson("regime", "loses in down/normal", ref=h.id, regime="down/normal")
    assert len(graph.lessons(h.id)) == 1


def test_graph_persists_across_reopen(tmp_path):
    path = str(tmp_path / "persist.sqlite")
    g = HypothesisGraph(path)
    h = g.add("durable idea")
    g.record(h.id, "refuted", {"sharpe": -0.3}, n_trials=7)
    g.close()

    again = HypothesisGraph(path)
    assert again.get(h.slug).status == "refuted"
    assert again.trial_count() == 7
    again.close()


def test_slugify_is_safe():
    assert slugify("A 20/100 MA cross!") == "a-20-100-ma-cross"
    assert slugify("!!!") == "hypothesis"
    assert len(slugify("x" * 200)) <= 60
