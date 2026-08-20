from src.analysis import (
    compare_predictors,
    worst_dialogues,
)
from src.benchmark import (
    DialogueBenchmarkResult,
    PredictorBenchmark,
)


def _benchmark(
    name: str,
    scores: dict[str, float],
) -> PredictorBenchmark:
    results = tuple(
        DialogueBenchmarkResult(
            dialogue_id=dialogue_id,
            n_chunks=5,
            prefix=(0, 1),
            score=score,
            prediction=[
                0,
                1,
                2,
                3,
                4,
            ],
            target=[
                0,
                1,
                2,
                3,
                4,
            ],
        )
        for dialogue_id, score
        in scores.items()
    )

    return PredictorBenchmark(
        predictor_name=name,
        mean_score=(
            100
            * sum(scores.values())
            / len(scores)
        ),
        dialogue_count=(
            len(scores)
        ),
        results=results,
    )


def test_worst_dialogues():
    benchmark = _benchmark(
        "test",
        {
            "0": 0.9,
            "1": 0.2,
            "2": 0.5,
        },
    )

    worst = worst_dialogues(
        benchmark,
        limit=2,
    )

    assert [
        result.dialogue_id
        for result in worst
    ] == [
        "1",
        "2",
    ]


def test_compare_predictors():
    baseline = _benchmark(
        "baseline",
        {
            "0": 0.5,
            "1": 0.5,
        },
    )

    candidate = _benchmark(
        "candidate",
        {
            "0": 0.8,
            "1": 0.3,
        },
    )

    deltas = compare_predictors(
        baseline=baseline,
        candidate=candidate,
    )

    assert deltas == [
        ("1", -0.2),
        ("0", 0.30000000000000004),
    ]