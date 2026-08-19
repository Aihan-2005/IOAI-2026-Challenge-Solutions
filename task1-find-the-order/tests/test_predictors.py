from pathlib import Path

from src.domain import Dialogue
from src.predictors import PrefixIndexBaseline


def _dialogue(
    *,
    n_chunks: int,
    prefix: tuple[int, int],
) -> Dialogue:
    return Dialogue(
        dialogue_id="0",
        directory=Path("/synthetic/0"),
        chunk_paths=tuple(
            Path(
                f"/synthetic/0/chunk_{index}.wav"
            )
            for index in range(n_chunks)
        ),
        prefix=prefix,
    )


def test_official_baseline_behavior():
    dialogue = _dialogue(
        n_chunks=6,
        prefix=(4, 1),
    )

    predictor = PrefixIndexBaseline()

    prediction = predictor.predict(dialogue)

    # Chronological order:
    # [4, 1, 0, 2, 3, 5]
    #
    # Therefore rank:
    # chunk 0 -> 2
    # chunk 1 -> 1
    # chunk 2 -> 3
    # chunk 3 -> 4
    # chunk 4 -> 0
    # chunk 5 -> 5

    assert prediction == [
        2,
        1,
        3,
        4,
        0,
        5,
    ]


def test_prefix_is_always_first_and_second():
    dialogue = _dialogue(
        n_chunks=8,
        prefix=(6, 3),
    )

    prediction = PrefixIndexBaseline().predict(
        dialogue
    )

    assert prediction[6] == 0
    assert prediction[3] == 1