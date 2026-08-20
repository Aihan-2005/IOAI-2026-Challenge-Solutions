from __future__ import annotations

from dataclasses import dataclass

from src.transition import TransitionMatrix


def _validate_prefix(
    *,
    n: int,
    prefix: tuple[int, int],
) -> None:
    first, second = prefix

    if n < 2:
        raise ValueError(
            "a dialogue must contain at least two chunks"
        )

    if first == second:
        raise ValueError(
            "prefix chunks must be different"
        )

    if not (
        0 <= first < n
        and 0 <= second < n
    ):
        raise ValueError(
            "prefix indexes are out of range"
        )


def path_score(
    order: list[int] | tuple[int, ...],
    transition_matrix: TransitionMatrix,
) -> float:
    """
    Sum direct-transition scores along a chronological path.

    Example:
        [1, 2, 0]

    score =
        score(1 -> 2)
        +
        score(2 -> 0)
    """
    if len(order) < 2:
        return 0.0

    return sum(
        transition_matrix.get(
            order[index],
            order[index + 1],
        )
        for index in range(len(order) - 1)
    )


def greedy_order(
    *,
    transition_matrix: TransitionMatrix,
    prefix: tuple[int, int],
) -> list[int]:
    """
    Greedy chronological ordering.

    At every step, select the highest-scoring next chunk.

    This is fast but locally optimal only.
    """
    n = transition_matrix.size

    _validate_prefix(
        n=n,
        prefix=prefix,
    )

    first, second = prefix

    order = [
        first,
        second,
    ]

    remaining = set(range(n))
    remaining.remove(first)
    remaining.remove(second)

    current = second

    while remaining:
        best_next = max(
            remaining,
            key=lambda candidate: (
                transition_matrix.get(
                    current,
                    candidate,
                ),
                -candidate,
            ),
        )

        order.append(best_next)

        remaining.remove(best_next)
        current = best_next

    return order


@dataclass(frozen=True, slots=True)
class _BeamState:
    order: tuple[int, ...]
    remaining: tuple[int, ...]
    score: float


def beam_search_order(
    *,
    transition_matrix: TransitionMatrix,
    prefix: tuple[int, int],
    beam_width: int = 32,
) -> list[int]:
    """
    Search for a high-scoring chronological ordering.

    Unlike greedy search, beam search keeps multiple candidate
    histories alive at every step.

    The known IOAI prefix is always preserved.
    """
    n = transition_matrix.size

    _validate_prefix(
        n=n,
        prefix=prefix,
    )

    if beam_width < 1:
        raise ValueError(
            "beam_width must be at least 1"
        )

    first, second = prefix

    initial_remaining = tuple(
        index
        for index in range(n)
        if index not in prefix
    )

    beam = [
        _BeamState(
            order=(first, second),
            remaining=initial_remaining,
            score=0.0,
        )
    ]

    while beam[0].remaining:
        candidates: list[_BeamState] = []

        for state in beam:
            current = state.order[-1]

            for next_chunk in state.remaining:
                transition_score = (
                    transition_matrix.get(
                        current,
                        next_chunk,
                    )
                )

                new_order = (
                    *state.order,
                    next_chunk,
                )

                new_remaining = tuple(
                    chunk
                    for chunk in state.remaining
                    if chunk != next_chunk
                )

                candidates.append(
                    _BeamState(
                        order=new_order,
                        remaining=new_remaining,
                        score=(
                            state.score
                            + transition_score
                        ),
                    )
                )

        candidates.sort(
            key=lambda state: (
                -state.score,
                state.order,
            )
        )


        beam = candidates[:beam_width]

    best = beam[0]

    return list(best.order)