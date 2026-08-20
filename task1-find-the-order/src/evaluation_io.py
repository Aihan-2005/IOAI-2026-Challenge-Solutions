from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from src.order_utils import (
    is_valid_permutation,
    pairwise_score,
)


class EvaluationFormatError(
    ValueError
):
    pass


def load_rank_answers(
    path: str | Path,
) -> dict[str, list[int]]:
    """
    Load and validate an IOAI answers JSON file.
    """
    answers_path = Path(path)

    if not answers_path.is_file():
        raise FileNotFoundError(
            f"Answers file not found: {answers_path}"
        )

    try:
        payload = json.loads(
            answers_path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as exc:
        raise EvaluationFormatError(
            f"Invalid JSON: {answers_path}"
        ) from exc

    if not isinstance(payload, dict):
        raise EvaluationFormatError(
            "Answers JSON root must be an object"
        )

    answers: dict[
        str,
        list[int],
    ] = {}

    for dialogue_id, rank in (
        payload.items()
    ):
        if not isinstance(
            dialogue_id,
            str,
        ):
            raise EvaluationFormatError(
                "Dialogue IDs must be strings"
            )

        if not is_valid_permutation(
            rank
        ):
            raise EvaluationFormatError(
                f"Dialogue {dialogue_id}: "
                "invalid rank permutation"
            )

        answers[dialogue_id] = rank

    return answers


def evaluate_subset(
    *,
    predictions: Mapping[
        str,
        object,
    ],
    targets: Mapping[
        str,
        list[int],
    ],
) -> dict[str, float]:
    """
    Return per-dialogue scores in the 0..1 scale.
    """
    scores: dict[
        str,
        float,
    ] = {}

    for dialogue_id, target in (
        targets.items()
    ):
        prediction = predictions.get(
            dialogue_id
        )

        scores[dialogue_id] = (
            pairwise_score(
                prediction,
                target,
            )
        )

    return scores


def macro_score(
    scores: Mapping[
        str,
        float,
    ],
) -> float:
    """
    Return macro-average score on the contest 0..100 scale.
    """
    if not scores:
        raise ValueError(
            "Cannot score an empty collection"
        )

    return (
        100.0
        * sum(scores.values())
        / len(scores)
    )