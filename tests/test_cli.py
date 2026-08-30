import json

import pytest

from sixth.cli import main


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "cli.sqlite")


def run(db, *args):
    return main(["--db", db, *args])


def test_init_creates_the_graph(db, capsys):
    assert run(db, "init") == 0
    assert "graph ready" in capsys.readouterr().out


def test_add_then_list(db, capsys):
    run(db, "add", "Momentum works in equities", "--tags", "trend,equity")
    capsys.readouterr()
    run(db, "list")
    out = capsys.readouterr().out
    assert "momentum-works-in-equities" in out
    assert "untested" in out


def test_add_blocks_near_duplicates_unless_forced(db, capsys):
    run(db, "add", "Momentum works in equities")
    capsys.readouterr()
    assert run(db, "add", "Momentum works in equities") == 2
    assert "already in the graph" in capsys.readouterr().out
    assert run(db, "add", "Momentum works in equities", "--force") == 0


def test_prereg_requires_a_falsifiable_condition(db, capsys):
    run(db, "add", "an idea", "--tags", "x")
    capsys.readouterr()
    assert run(db, "prereg", "an-idea") == 2
    assert "falsifiable" in capsys.readouterr().err


def test_prereg_rejects_malformed_conditions(db, capsys):
    run(db, "add", "an idea")
    capsys.readouterr()
    assert run(db, "prereg", "an-idea", "--must", "sharpe is great") == 2
    assert "comparison operator" in capsys.readouterr().err


def test_run_without_a_prereg_is_refused(db, capsys):
    run(db, "add", "an idea")
    capsys.readouterr()
    assert run(db, "run", "an-idea", "--strategy", "sma_cross", "--bars", "800") == 2
    assert "preregistration" in capsys.readouterr().err


def test_run_on_an_unknown_slug_is_refused(db, capsys):
    assert run(db, "run", "nope", "--strategy", "sma_cross") == 2
    assert "no hypothesis" in capsys.readouterr().err


def test_full_cli_cycle(db, capsys):
    run(db, "add", "A moving average cross beats buy and hold", "--tags", "trend")
    run(db, "prereg", "a-moving-average-cross-beats-buy-and-hold",
        "--must", "sharpe >= 0.8", "--must", "max_drawdown <= 0.25",
        "--rationale", "trends persist")
    capsys.readouterr()
    assert run(db, "run", "a-moving-average-cross-beats-buy-and-hold",
               "--strategy", "sma_cross", "--grid", "fast=10,20", "slow=60,100",
               "--bars", "1200", "--folds", "3", "--null-samples", "20") == 0
    out = capsys.readouterr().out
    assert "CYCLE COMPLETE" in out
    assert "SEALED HOLDOUT" in out
    assert "WRITTEN TO GRAPH" in out

    run(db, "status")
    status = capsys.readouterr().out
    assert "experiments 1" in status.replace("  ", " ")

    run(db, "context")
    assert "Prior research state" in capsys.readouterr().out


def test_run_json_output_is_parseable(db, capsys):
    run(db, "add", "an idea")
    run(db, "prereg", "an-idea", "--must", "sharpe >= 0.5")
    capsys.readouterr()
    run(db, "run", "an-idea", "--strategy", "momentum", "--bars", "1000",
        "--folds", "3", "--null-samples", "10", "--json")
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"]
    assert payload["walkforward"]["n_folds"] == 3


def test_context_json(db, capsys):
    run(db, "add", "an idea")
    capsys.readouterr()
    run(db, "context", "--json")
    assert "frontier" in json.loads(capsys.readouterr().out)


def test_frontier_and_lessons_on_an_empty_graph(db, capsys):
    run(db, "init")
    capsys.readouterr()
    run(db, "frontier")
    assert "frontier is empty" in capsys.readouterr().out
    run(db, "lessons")
    assert "no lessons" in capsys.readouterr().out


def test_grid_parsing_rejects_bad_syntax(db):
    run(db, "add", "an idea")
    run(db, "prereg", "an-idea", "--must", "sharpe >= 0.5")
    with pytest.raises(SystemExit):
        run(db, "run", "an-idea", "--strategy", "sma_cross", "--grid", "fastis10")


def test_strategies_lists_the_registry(db, capsys):
    run(db, "strategies")
    out = capsys.readouterr().out
    for name in ("sma_cross", "momentum", "mean_reversion", "buy_hold"):
        assert name in out


def test_link_and_show(db, capsys):
    run(db, "add", "base idea")
    run(db, "add", "refined idea")
    capsys.readouterr()
    run(db, "link", "refined-idea", "base-idea", "--kind", "refines")
    run(db, "show", "refined-idea")
    assert "base-idea" in capsys.readouterr().out


def test_export_is_valid_json(db, capsys):
    run(db, "add", "an idea")
    capsys.readouterr()
    run(db, "export")
    data = json.loads(capsys.readouterr().out)
    assert data["hypotheses"][0]["slug"] == "an-idea"


def test_demo_runs_end_to_end(db, capsys):
    assert main(["--db", db, "demo", "--bars", "1000"]) == 0
    out = capsys.readouterr().out
    assert "CYCLE COMPLETE" in out
    assert "Prior research state" in out
