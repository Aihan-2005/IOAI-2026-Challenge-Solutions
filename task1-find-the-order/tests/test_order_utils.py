import pytest

from src.order_utils import (
    is_valid_permutation,
    order_to_rank,
    pairwise_score,
    rank_to_order,
    submission_score,
)


def test_order_to_rank():
    order = [1, 2, 0]

    assert order_to_rank(order) == [2, 0, 1]


def test_rank_to_order():
    rank = [2, 0, 1]

    assert rank_to_order(rank) == [1, 2, 0]


def test_round_trip_conversion():
    order = [4, 1, 0, 3, 2]

    rank = order_to_rank(order)

    assert rank_to_order(rank) == order


def test_valid_permutation():
    assert is_valid_permutation([2, 0, 1])

    assert not is_valid_permutation([0, 0, 1])
    assert not is_valid_permutation([0, 1, 3])
    assert not is_valid_permutation([0, True, 2])
    assert not is_valid_permutation((0, 1, 2))


def test_invalid_order_raises():
    with pytest.raises(ValueError):
        order_to_rank([0, 0, 1])


def test_perfect_pairwise_score():
    truth = [2, 0, 1]

    assert pairwise_score(truth, truth) == 1.0


def test_completely_reversed_order_scores_zero():
    truth_order = [1, 2, 0]
    truth_rank = order_to_rank(truth_order)

    reversed_order = list(reversed(truth_order))
    reversed_rank = order_to_rank(reversed_order)

    assert pairwise_score(reversed_rank, truth_rank) == 0.0


def test_invalid_prediction_scores_zero():
    truth = [2, 0, 1]

    assert pairwise_score([0, 0, 1], truth) == 0.0
    assert pairwise_score(None, truth) == 0.0


def test_submission_score_uses_macro_average():
    targets = {
        "dialogue_a": [2, 0, 1],
        "dialogue_b": [0, 1, 2],
    }

    predictions = {
        "dialogue_a": [2, 0, 1],
    }

    assert submission_score(predictions, targets) == 50.0