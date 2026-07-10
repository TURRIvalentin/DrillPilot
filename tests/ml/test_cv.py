"""Tests for the leave-one-well-out CV splitter (ml.training.cv)."""

from __future__ import annotations

import pandas as pd

from ml.training.cv import leave_one_well_out_splits


def test_folds_cover_all_rows_exactly_once_as_validation() -> None:
    well_id = pd.Series([0, 0, 1, 1, 1, 2, 2])

    seen_val: set[int] = set()
    for _train_pos, val_pos, _held_out in leave_one_well_out_splits(well_id):
        seen_val.update(val_pos.tolist())

    assert seen_val == set(range(len(well_id)))


def test_train_never_includes_the_held_out_well() -> None:
    well_id = pd.Series([0, 0, 1, 1, 1, 2, 2])

    for train_pos, val_pos, held_out in leave_one_well_out_splits(well_id):
        assert (well_id.iloc[train_pos] == held_out).sum() == 0
        assert (well_id.iloc[val_pos] == held_out).all()


def test_number_of_folds_matches_distinct_wells() -> None:
    well_id = pd.Series([5, 5, 7, 9, 9, 9])

    folds = list(leave_one_well_out_splits(well_id))

    assert len(folds) == 3
    assert {held_out for _, _, held_out in folds} == {5, 7, 9}


def test_positions_are_positional_not_label_based() -> None:
    well_id = pd.Series([0, 0, 1, 1], index=[100, 101, 102, 103])

    for train_pos, val_pos, _held_out in leave_one_well_out_splits(well_id):
        assert max(train_pos.tolist() + val_pos.tolist()) < len(well_id)
