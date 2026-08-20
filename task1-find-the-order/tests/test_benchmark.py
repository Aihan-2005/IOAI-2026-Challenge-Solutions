from pathlib import Path

from src.benchmark import (
    build_report,
    evaluate_predictor,
)
from src.domain import Dialogue
from src.predictors import (
    PrefixIndexBaseline,
)


def _dialogue(
    dialogue_id: str,
    prefix: tuple[int, int],
    n_chunks: int,
) -> Dialogue:
    return Dialogue(
        dialogue_id=dialogue_id,
        directory=Path(
            f"/synthetic/{dialogue_id}"
        ),
        chunk_paths=tuple(
            Path(
                f"/synthetic/"
                f"{dialogue_id}/"
                f"chunk_{index}.wav"
            )
            for index in range(
                n_chunks
            )
        ),
        prefix=prefix,
    )


def test_evaluate_predictor():
    dialogues = (
        _dialogue(
            "0",
            (1, 2),
            3,
        ),
    )

    targets = {
        "0": [2, 0, 1],
    }

    benchmark = evaluate_predictor(
        dialogues=dialogues,
        targets=targets,
        predictor=(
            PrefixIndexBaseline()
        ),
    )

    assert (
        benchmark.dialogue_count
        == 1
    )

    assert (
        benchmark.mean_score
        == 100.0
    )


def test_build_report():
    dialogues = (
        _dialogue(
            "0",
            (1, 2),
            3,
        ),
    )

    benchmark = evaluate_predictor(
        dialogues=dialogues,
        targets={
            "0": [2, 0, 1],
        },
        predictor=(
            PrefixIndexBaseline()
        ),
    )

    report = build_report(
        dataset_name="synthetic",
        benchmarks=(
            benchmark,
        ),
    )

    assert (
        report.dataset_name
        == "synthetic"
    )

    assert (
        len(report.predictors)
        == 1
    )