import pytest

from sixth.backtest import Costs
from sixth.walkforward import (expand_grid, purged_splits, seal_holdout,
                               walk_forward)


def test_seal_holdout_splits_without_overlap(bars):
    dev, sealed = seal_holdout(bars, 0.2)
    assert len(dev) + len(sealed) == len(bars)
    assert dev.dates[-1] < sealed.dates[0]
    assert len(sealed) == pytest.approx(len(bars) * 0.2, abs=1)


def test_seal_holdout_rejects_bad_fraction(bars):
    for bad in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(ValueError):
            seal_holdout(bars, bad)


def test_splits_respect_min_train_and_embargo():
    splits = purged_splits(2000, n_folds=5, embargo=10, min_train=250)
    assert len(splits) == 5
    for (tr0, tr1), (te0, te1) in splits:
        assert tr1 - tr0 >= 250, "training window shorter than min_train"
        assert te0 - tr1 == 10, "embargo gap not honoured"
        assert te1 > te0


def test_splits_are_ordered_and_cover_the_tail():
    splits = purged_splits(2000, n_folds=4, embargo=5, min_train=300)
    tests = [t for _, t in splits]
    assert tests[-1][1] == 2000
    for a, b in zip(tests, tests[1:]):
        assert a[1] <= b[0], "test windows must not overlap"


def test_training_never_includes_its_own_test_window():
    for (tr0, tr1), (te0, _) in purged_splits(3000, 6, 20, 250):
        assert tr1 <= te0 - 20


def test_too_little_data_raises():
    with pytest.raises(ValueError):
        purged_splits(260, n_folds=5, embargo=10, min_train=250)


def test_expand_grid_is_a_full_product():
    grid = expand_grid({"a": [1, 2], "b": [3, 4, 5]})
    assert len(grid) == 6
    assert {"a": 1, "b": 3} in grid


def test_expand_grid_of_nothing_is_one_empty_run():
    assert expand_grid({}) == [{}]


def test_trials_counts_every_candidate_on_every_fold(bars):
    dev, _ = seal_holdout(bars, 0.2)
    grid = {"fast": [10, 20], "slow": [50, 100, 200]}
    wf = walk_forward("sma_cross", dev, grid, Costs(), n_folds=3)
    assert wf.n_trials == 6 * 3
    assert wf.metrics["n_trials"] == 18.0


def test_deflated_sharpe_is_reported_against_the_trial_count(bars):
    dev, _ = seal_holdout(bars, 0.2)
    wf = walk_forward("sma_cross", dev, {"fast": [5, 10, 20, 40],
                                         "slow": [50, 100, 150, 200]},
                      Costs(), n_folds=3)
    assert 0.0 <= wf.metrics["dsr"] <= 1.0
    assert wf.metrics["n_trials"] == 48.0


def test_oos_returns_line_up_with_dates(bars):
    dev, _ = seal_holdout(bars, 0.2)
    wf = walk_forward("momentum", dev, {"lookback": [20, 60]}, Costs(), n_folds=3)
    assert len(wf.oos_returns) == len(wf.oos_dates)
    assert len(set(wf.oos_dates)) == len(wf.oos_dates), "a bar was scored twice"


def test_param_stability_is_a_fraction(bars):
    dev, _ = seal_holdout(bars, 0.2)
    wf = walk_forward("sma_cross", dev, {"fast": [10, 20]}, Costs(), n_folds=4)
    assert 0.0 < wf.metrics["param_stability"] <= 1.0
