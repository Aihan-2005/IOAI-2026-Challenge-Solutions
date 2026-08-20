from __future__ import annotations

from src.benchmark import (
    PredictorBenchmark,
)


def worst_dialogues(
    benchmark: PredictorBenchmark,
    *,
    limit: int = 10,
) -> tuple:
    """
    Return lowest-scoring dialogues first.
    """
    if limit < 1:
        raise ValueError(
            "limit must be positive"
        )

    return tuple(
        sorted(
            benchmark.results,
            key=lambda result: (
                result.score,
                -result.n_chunks,
                int(result.dialogue_id),
            ),
        )[:limit]
    )


def compare_predictors(
    *,
    baseline: PredictorBenchmark,
    candidate: PredictorBenchmark,
) -> list[
    tuple[str, float]
]:
    """
    Return per-dialogue score delta:

        candidate - baseline
    """
    baseline_scores = {
        result.dialogue_id: result.score
        for result in baseline.results
    }

    deltas: list[
        tuple[str, float]
    ] = []

    for result in candidate.results:
        if (
            result.dialogue_id
            not in baseline_scores
        ):
            continue

        delta = (
            result.score
            - baseline_scores[
                result.dialogue_id
            ]
        )

        deltas.append(
            (
                result.dialogue_id,
                delta,
            )
        )

    return sorted(
        deltas,
        key=lambda item: (
            item[1],
            int(item[0]),
        ),
    )