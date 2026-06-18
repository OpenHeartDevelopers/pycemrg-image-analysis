# tests/unit/test_metrics_dice.py

"""Unit tests for Dice coefficient metrics in utilities/metrics.py."""

import math
import numpy as np
import pytest

from pycemrg_image_analysis.utilities.metrics import (
    compute_dice,
    compute_dice_per_label,
)


# ---------------------------------------------------------------------------
# compute_dice
# ---------------------------------------------------------------------------


def test_identical_masks_score_one():
    mask = np.array([[0, 1], [1, 1]])
    assert compute_dice(mask, mask) == 1.0


def test_disjoint_masks_score_zero():
    pred = np.array([[1, 0], [0, 0]])
    gt = np.array([[0, 1], [0, 0]])
    assert compute_dice(pred, gt) == 0.0


def test_half_overlap_known_value():
    # |A|=3, |B|=2, intersection=2 -> 2*2/(3+2) = 0.8
    pred = np.array([0, 1, 1, 1])
    gt = np.array([0, 1, 1, 0])
    assert compute_dice(pred, gt) == pytest.approx(0.8)


def test_bool_and_int_inputs_agree():
    pred_int = np.array([0, 1, 1, 0])
    gt_int = np.array([0, 1, 0, 0])
    pred_bool = pred_int.astype(bool)
    gt_bool = gt_int.astype(bool)
    assert compute_dice(pred_int, gt_int) == compute_dice(pred_bool, gt_bool)


def test_both_empty_returns_nan():
    empty = np.zeros((3, 3), dtype=int)
    assert math.isnan(compute_dice(empty, empty))


def test_one_empty_returns_zero():
    empty = np.zeros((3, 3), dtype=int)
    present = np.ones((3, 3), dtype=int)
    assert compute_dice(present, empty) == 0.0
    assert compute_dice(empty, present) == 0.0


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_dice(np.zeros((2, 2)), np.zeros((3, 3)))


# ---------------------------------------------------------------------------
# compute_dice_per_label
# ---------------------------------------------------------------------------


def test_per_label_returns_correct_dict():
    pred = np.array([0, 1, 1, 2, 2])
    gt = np.array([0, 1, 2, 2, 2])
    scores = compute_dice_per_label(pred, gt)
    # label 1: |A|=2,|B|=1,inter=1 -> 2/3 ; label 2: |A|=2,|B|=3,inter=2 -> 0.8
    assert set(scores.keys()) == {1, 2}
    assert scores[1] == pytest.approx(2 / 3)
    assert scores[2] == pytest.approx(0.8)


def test_background_excluded_by_default():
    pred = np.array([0, 0, 1, 1])
    gt = np.array([0, 0, 1, 1])
    scores = compute_dice_per_label(pred, gt)
    assert 0 not in scores
    assert scores == {1: 1.0}


def test_background_included_with_flag():
    pred = np.array([0, 0, 1, 1])
    gt = np.array([0, 0, 1, 1])
    scores = compute_dice_per_label(pred, gt, include_background=True)
    assert scores[0] == 1.0
    assert scores[1] == 1.0


def test_explicit_label_subset_honoured():
    pred = np.array([0, 1, 2, 3])
    gt = np.array([0, 1, 2, 3])
    scores = compute_dice_per_label(pred, gt, labels=[2])
    assert set(scores.keys()) == {2}
    assert scores[2] == 1.0


def test_background_only_image_returns_empty_dict():
    background = np.zeros((4, 4), dtype=int)
    assert compute_dice_per_label(background, background) == {}


def test_keys_are_plain_ints():
    pred = np.array([0, 1, 1], dtype=np.int64)
    gt = np.array([0, 1, 1], dtype=np.int64)
    scores = compute_dice_per_label(pred, gt)
    assert all(type(key) is int for key in scores.keys())


def test_nanmean_composition_ignores_absent_label():
    # label 3 absent from both -> NaN, must not poison the mean
    pred = np.array([1, 1, 2, 2])
    gt = np.array([1, 1, 2, 2])
    scores = compute_dice_per_label(pred, gt, labels=[1, 2, 3])
    assert math.isnan(scores[3])
    mean_dice = float(np.nanmean(list(scores.values())))
    assert mean_dice == pytest.approx(1.0)


def test_per_label_shape_mismatch_raises():
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_dice_per_label(np.zeros((2, 2)), np.zeros((3, 3)))
