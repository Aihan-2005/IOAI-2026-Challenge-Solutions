from pathlib import Path

from src.dialogue_selection import (
    deterministic_dialogue_sample,
)
from src.domain import Dialogue


def _dialogue(
    dialogue_id: str,
) -> Dialogue:
    return Dialogue(
        dialogue_id=dialogue_id,
        directory=Path(
            f"/synthetic/{dialogue_id}"
        ),
        chunk_paths=(
            Path(
                f"/synthetic/"
                f"{dialogue_id}/"
                "chunk_0.wav"
            ),
            Path(
                f"/synthetic/"
                f"{dialogue_id}/"
                "chunk_1.wav"
            ),
        ),
        prefix=(0, 1),
    )


def test_selection_is_reproducible():
    dialogues = tuple(
        _dialogue(
            str(index)
        )
        for index in range(20)
    )

    first = (
        deterministic_dialogue_sample(
            dialogues,
            count=5,
            seed="test-seed",
        )
    )

    second = (
        deterministic_dialogue_sample(
            dialogues,
            count=5,
            seed="test-seed",
        )
    )

    assert [
        item.dialogue_id
        for item in first
    ] == [
        item.dialogue_id
        for item in second
    ]


def test_selection_has_requested_size():
    dialogues = tuple(
        _dialogue(
            str(index)
        )
        for index in range(20)
    )

    selected = (
        deterministic_dialogue_sample(
            dialogues,
            count=7,
            seed="test-seed",
        )
    )

    assert len(selected) == 7

    assert (
        len(
            {
                dialogue.dialogue_id
                for dialogue in selected
            }
        )
        == 7
    )