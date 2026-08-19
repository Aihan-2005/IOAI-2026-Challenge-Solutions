from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from src.domain import Dialogue
from src.order_utils import is_valid_permutation


class SubmissionError(ValueError):
    """Raised when a submission is structurally invalid."""


def validate_submission(
    dialogues: tuple[Dialogue, ...],
    predictions: Mapping[str, object],
) -> None:
    expected_ids = {
        dialogue.dialogue_id
        for dialogue in dialogues
    }

    predicted_ids = set(predictions)

    missing = expected_ids - predicted_ids
    extra = predicted_ids - expected_ids

    if missing:
        raise SubmissionError(
            f"Missing predictions for dialogues: "
            f"{sorted(missing, key=int)}"
        )

    if extra:
        raise SubmissionError(
            f"Unknown dialogue IDs in predictions: "
            f"{sorted(extra, key=int)}"
        )

    for dialogue in dialogues:
        prediction = predictions[dialogue.dialogue_id]

        if not is_valid_permutation(
            prediction,
            dialogue.n_chunks,
        ):
            raise SubmissionError(
                f"Dialogue {dialogue.dialogue_id}: "
                f"invalid prediction {prediction!r}"
            )


def write_submission(
    output_path: str | Path,
    dialogues: tuple[Dialogue, ...],
    predictions: Mapping[str, list[int]],
) -> Path:
    """
    Validate and atomically write answers.json.
    """
    validate_submission(
        dialogues=dialogues,
        predictions=predictions,
    )

    path = Path(output_path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    serialized = json.dumps(
        dict(predictions),
        ensure_ascii=False,
        separators=(",", ":"),
    )

    temporary_path.write_text(
        serialized,
        encoding="utf-8",
    )

    temporary_path.replace(path)

    return path


def read_submission(
    path: str | Path,
) -> dict[str, object]:
    submission_path = Path(path)

    try:
        data = json.loads(
            submission_path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as exc:
        raise SubmissionError(
            f"Invalid JSON submission: {submission_path}"
        ) from exc

    if not isinstance(data, dict):
        raise SubmissionError(
            "Submission root must be a JSON object"
        )

    return data