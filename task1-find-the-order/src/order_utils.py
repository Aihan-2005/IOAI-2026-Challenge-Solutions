from __future__ import annotations

from collections.abc import Mapping, Sequence


def is_valid_permutation(values: object, n: int | None = None) -> bool:
    """
    Return True iff `values` is a valid permutation of 0..n-1.

    This intentionally mirrors the IOAI grader behavior:
    - must be a list
    - correct length
    - integers only
    - bool is rejected even though bool is a subclass of int
    - no duplicates
    - no out-of-range values
    """
    if not isinstance(values, list):
        return False

    expected_n = len(values) if n is None else n

    if len(values) != expected_n:
        return False

    seen = [False] * expected_n

    for value in values:
        if not isinstance(value, int) or isinstance(value, bool):
            return False

        if value < 0 or value >= expected_n:
            return False

        if seen[value]:
            return False

        seen[value] = True

    return True


def order_to_rank(order: Sequence[int]) -> list[int]:
    """
    Convert chronological chunk order to IOAI rank convention.

    Example
    -------
    Chronological order:
        chunk_1 -> chunk_2 -> chunk_0

    Input:
        [1, 2, 0]

    Output:
        [2, 0, 1]

    Meaning:
        chunk_0 is at position 2
        chunk_1 is at position 0
        chunk_2 is at position 1
    """
    order_list = list(order)

    if not is_valid_permutation(order_list):
        raise ValueError("order must be a permutation of 0..n-1")

    rank = [0] * len(order_list)

    for position, chunk_idx in enumerate(order_list):
        rank[chunk_idx] = position

    return rank


def rank_to_order(rank: Sequence[int]) -> list[int]:
    """
    Convert IOAI rank convention to chronological chunk order.
    """
    rank_list = list(rank)

    if not is_valid_permutation(rank_list):
        raise ValueError("rank must be a permutation of 0..n-1")

    return sorted(
        range(len(rank_list)),
        key=lambda chunk_idx: rank_list[chunk_idx],
    )


def pairwise_score(
    predicted_rank: object,
    true_rank: Sequence[int],
) -> float:
    """
    Compute the official per-dialogue pairwise ordering accuracy.

    Returns a value in [0, 1].

    Invalid predictions receive 0.0, matching the contest grader.
    """
    target = list(true_rank)
    n = len(target)

    if not is_valid_permutation(target):
        raise ValueError("true_rank must be a valid permutation")

    if n < 2:
        return 1.0

    if not is_valid_permutation(predicted_rank, n):
        return 0.0

    prediction = predicted_rank
    discordant_pairs = 0
    total_pairs = n * (n - 1) // 2

    for i in range(n):
        for j in range(i + 1, n):
            true_relation = target[i] > target[j]
            predicted_relation = prediction[i] > prediction[j]

            if true_relation != predicted_relation:
                discordant_pairs += 1

    return 1.0 - discordant_pairs / total_pairs


def submission_score(
    predictions: Mapping[str, object],
    targets: Mapping[str, Sequence[int]],
) -> float:
    """
    Compute the contest-style macro score in the 0..100 scale.

    Missing dialogue predictions automatically receive 0.
    """
    if not targets:
        raise ValueError("targets cannot be empty")

    scores = [
        pairwise_score(
            predictions.get(dialogue_id),
            target_rank,
        )
        for dialogue_id, target_rank in targets.items()
    ]

    mean_score = sum(scores) / len(scores)

    return 100.0 * mean_score