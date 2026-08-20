from src.global_ordering import greedy_order
from src.transition import TransitionMatrix


def test_greedy_order_respects_prefix():
    matrix = TransitionMatrix(
        scores=(
            (
                float("-inf"),
                0.0,
                0.0,
                0.0,
            ),
            (
                0.0,
                float("-inf"),
                10.0,
                1.0,
            ),
            (
                0.0,
                0.0,
                float("-inf"),
                9.0,
            ),
            (
                0.0,
                0.0,
                0.0,
                float("-inf"),
            ),
        )
    )

    order = greedy_order(
        transition_matrix=matrix,
        prefix=(0, 1),
    )

    assert order == [
        0,
        1,
        2,
        3,
    ]


def test_greedy_order_returns_permutation():
    matrix = TransitionMatrix(
        scores=(
            (
                float("-inf"),
                2.0,
                1.0,
            ),
            (
                1.0,
                float("-inf"),
                4.0,
            ),
            (
                3.0,
                1.0,
                float("-inf"),
            ),
        )
    )

    order = greedy_order(
        transition_matrix=matrix,
        prefix=(0, 1),
    )

    assert sorted(order) == [
        0,
        1,
        2,
    ]

    assert order[:2] == [
        0,
        1,
    ]