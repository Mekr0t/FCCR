"""Toy-data tests for src/checks.assert_predictions_in_pool (no dataset needed)."""

import pytest

from src.checks import assert_predictions_in_pool


def _units():
    return [
        {"lang": "eng", "post_ids": [1, 2], "pool_fc_ids": list(range(100, 115))},
        {"lang": "spa", "post_ids": [3], "pool_fc_ids": list(range(200, 212))},
    ]


def _good():
    return {
        1: list(range(100, 110)),
        2: list(range(105, 115)),
        3: list(range(200, 210)),
    }


def test_valid_predictions_pass():
    assert assert_predictions_in_pool(_good(), _units()) is None


def test_str_and_int_ids_are_interchangeable():
    preds = {"1": [str(x) for x in range(100, 110)], "2": list(range(105, 115)), "3": list(range(200, 210))}
    assert_predictions_in_pool(preds, _units())


def test_id_from_another_language_pool_is_rejected():
    preds = _good()
    preds[3] = [100] + list(range(200, 209))  # 100 belongs to the eng pool, not the spa pool
    with pytest.raises(ValueError, match=r"\[spa\] post 3: 1 ids outside the pool"):
        assert_predictions_in_pool(preds, _units())


def test_fewer_than_ten_predictions_is_rejected():
    preds = _good()
    preds[2] = list(range(105, 114))  # 9 ids
    with pytest.raises(ValueError, match=r"post 2: 9 predictions, need >= 10"):
        assert_predictions_in_pool(preds, _units())


def test_duplicate_ids_are_rejected():
    preds = _good()
    preds[1] = [100] * 10
    with pytest.raises(ValueError, match=r"post 1: duplicated fact-check ids"):
        assert_predictions_in_pool(preds, _units())


def test_missing_post_is_rejected():
    preds = _good()
    del preds[3]
    with pytest.raises(ValueError, match=r"post 3: no predictions"):
        assert_predictions_in_pool(preds, _units())


def test_extra_post_is_rejected():
    preds = _good()
    preds[99] = list(range(100, 110))
    with pytest.raises(ValueError, match=r"not evaluated, e.g. \['99'\]"):
        assert_predictions_in_pool(preds, _units())


def test_crosslingual_pool_accepts_any_language_fc():
    units = [{"lang": "cross", "post_ids": [1, 2], "pool_fc_ids": list(range(100, 130))}]
    preds = {1: list(range(100, 110)), 2: list(range(120, 130))}
    assert_predictions_in_pool(preds, units)


def test_error_lists_at_most_ten_problems():
    units = [{"lang": "eng", "post_ids": list(range(20)), "pool_fc_ids": [1]}]
    with pytest.raises(ValueError, match=r"\(\+10 more\)"):
        assert_predictions_in_pool({}, units)
