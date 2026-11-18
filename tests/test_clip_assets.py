"""Calibration budgets: how C_t and H_t are placed inside a calibration draw."""

import pytest

from if_merge.clip.assets import budget_sizes, calibration_split


def test_default_budget_is_256_plus_256():
    assert budget_sizes(512) == (256, 256)
    statistics, holdout = calibration_split(256, 256)
    assert statistics == list(range(256)) and holdout == list(range(256, 512))


def test_small_budgets_are_halved():
    assert budget_sizes(128) == (64, 64)
    assert calibration_split(64, 64) == (list(range(64)), list(range(64, 128)))
    with pytest.raises(ValueError):
        budget_sizes(127)


def test_large_budgets_keep_the_default_holdout_set_and_are_nested():
    assert budget_sizes(2048) == (1792, 256)
    statistics, holdout = calibration_split(*budget_sizes(2048))
    assert holdout == list(range(256, 512))
    assert statistics == list(range(256)) + list(range(512, 2048))
    smaller, _ = calibration_split(*budget_sizes(1024))
    assert set(smaller) < set(statistics) and not set(statistics) & set(holdout)
