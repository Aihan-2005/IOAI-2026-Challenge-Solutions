import json

import pytest

from src.dataset_io import (
    DatasetFormatError,
    load_dialogues,
)


def _create_chunks(
    directory,
    count: int,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for index in range(count):
        (
            directory / f"chunk_{index}.wav"
        ).write_bytes(b"")


def test_load_dialogues_from_ioai_structure(
    tmp_path,
):
    split = tmp_path / "test_public"
    split.mkdir()

    (split / "prefix.json").write_text(
        json.dumps(
            {
                "0": [1, 2],
                "12": [2, 0],
            }
        )
    )

    _create_chunks(split / "0", 4)
    _create_chunks(split / "12", 3)

    # Must be ignored.
    (split / ".ipynb_checkpoints").mkdir()

    dialogues = load_dialogues(split)

    assert [
        dialogue.dialogue_id
        for dialogue in dialogues
    ] == ["0", "12"]

    assert dialogues[0].prefix == (1, 2)
    assert dialogues[0].n_chunks == 4

    assert dialogues[0].chunk_paths[0].name == (
        "chunk_0.wav"
    )

    assert dialogues[1].prefix == (2, 0)
    assert dialogues[1].n_chunks == 3


def test_missing_prefix_entry_is_rejected(
    tmp_path,
):
    split = tmp_path / "test_public"
    split.mkdir()

    (split / "prefix.json").write_text(
        json.dumps({})
    )

    _create_chunks(split / "0", 3)

    with pytest.raises(
        DatasetFormatError,
        match="missing from prefix.json",
    ):
        load_dialogues(split)


def test_non_contiguous_chunks_are_rejected(
    tmp_path,
):
    split = tmp_path / "test_public"
    split.mkdir()

    (split / "prefix.json").write_text(
        json.dumps(
            {
                "0": [0, 2],
            }
        )
    )

    dialogue = split / "0"
    dialogue.mkdir()

    (dialogue / "chunk_0.wav").write_bytes(b"")
    (dialogue / "chunk_2.wav").write_bytes(b"")

    with pytest.raises(
        DatasetFormatError,
        match="must be contiguous",
    ):
        load_dialogues(split)