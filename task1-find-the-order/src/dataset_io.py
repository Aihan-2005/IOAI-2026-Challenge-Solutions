from __future__ import annotations

import json
import re
from pathlib import Path

from src.domain import Dialogue


_CHUNK_PATTERN = re.compile(r"^chunk_(\d+)\.wav$")


class DatasetFormatError(ValueError):
    """Raised when an IOAI dataset split is malformed."""


def _read_json(path: Path) -> object:
    if not path.is_file():
        raise DatasetFormatError(
            f"Required file does not exist: {path}"
        )

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetFormatError(
            f"Invalid JSON file: {path}"
        ) from exc


def load_prefixes(
    split_dir: str | Path,
) -> dict[str, tuple[int, int]]:
    """
    Load prefix.json.

    Returns:
        {
            dialogue_id: (first_chunk_index, second_chunk_index)
        }
    """
    split_path = Path(split_dir)
    raw = _read_json(split_path / "prefix.json")

    if not isinstance(raw, dict):
        raise DatasetFormatError(
            "prefix.json must contain a JSON object"
        )

    prefixes: dict[str, tuple[int, int]] = {}

    for dialogue_id, value in raw.items():
        if not isinstance(dialogue_id, str):
            raise DatasetFormatError(
                "dialogue IDs in prefix.json must be strings"
            )

        if (
            not isinstance(value, list)
            or len(value) != 2
        ):
            raise DatasetFormatError(
                f"Dialogue {dialogue_id}: "
                "prefix must contain exactly two indexes"
            )

        first, second = value

        if (
            not isinstance(first, int)
            or isinstance(first, bool)
            or not isinstance(second, int)
            or isinstance(second, bool)
        ):
            raise DatasetFormatError(
                f"Dialogue {dialogue_id}: "
                "prefix indexes must be integers"
            )

        if first == second:
            raise DatasetFormatError(
                f"Dialogue {dialogue_id}: "
                "prefix indexes must be different"
            )

        prefixes[dialogue_id] = (first, second)

    return prefixes


def discover_chunk_paths(
    dialogue_dir: str | Path,
) -> tuple[Path, ...]:
    """
    Discover chunk_0.wav ... chunk_{n-1}.wav.

    The returned tuple is indexed directly by shuffled chunk index.
    """
    dialogue_path = Path(dialogue_dir)

    indexed_paths: dict[int, Path] = {}

    for path in dialogue_path.iterdir():
        if not path.is_file():
            continue

        match = _CHUNK_PATTERN.fullmatch(path.name)

        if match is None:
            continue

        index = int(match.group(1))
        indexed_paths[index] = path

    if not indexed_paths:
        raise DatasetFormatError(
            f"No chunk_*.wav files found in {dialogue_path}"
        )

    expected_indexes = list(range(len(indexed_paths)))
    actual_indexes = sorted(indexed_paths)

    if actual_indexes != expected_indexes:
        raise DatasetFormatError(
            f"Chunks in {dialogue_path} must be contiguous "
            f"0..{len(indexed_paths) - 1}; "
            f"found {actual_indexes}"
        )

    return tuple(
        indexed_paths[index]
        for index in expected_indexes
    )


def load_dialogues(
    split_dir: str | Path,
) -> tuple[Dialogue, ...]:
    """
    Load all numeric dialogue directories from an IOAI split.

    Important:
    This function never reads answers.json.
    It is therefore safe for hidden-test inference.
    """
    split_path = Path(split_dir)

    if not split_path.is_dir():
        raise DatasetFormatError(
            f"Dataset split does not exist: {split_path}"
        )

    prefixes = load_prefixes(split_path)

    dialogue_directories = sorted(
        (
            path
            for path in split_path.iterdir()
            if path.is_dir() and path.name.isdigit()
        ),
        key=lambda path: int(path.name),
    )

    dialogues: list[Dialogue] = []

    for directory in dialogue_directories:
        dialogue_id = directory.name

        if dialogue_id not in prefixes:
            raise DatasetFormatError(
                f"Dialogue {dialogue_id} is missing "
                "from prefix.json"
            )

        chunk_paths = discover_chunk_paths(directory)
        prefix = prefixes[dialogue_id]

        n_chunks = len(chunk_paths)
        first, second = prefix

        if (
            first < 0
            or second < 0
            or first >= n_chunks
            or second >= n_chunks
        ):
            raise DatasetFormatError(
                f"Dialogue {dialogue_id}: "
                f"prefix {prefix} is invalid for "
                f"{n_chunks} chunks"
            )

        dialogues.append(
            Dialogue(
                dialogue_id=dialogue_id,
                directory=directory,
                chunk_paths=chunk_paths,
                prefix=prefix,
            )
        )

    return tuple(dialogues)