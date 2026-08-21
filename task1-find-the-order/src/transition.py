from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from src.transcription import DialogueTranscript


TransitionPair = tuple[str, str]


class TransitionScorer(Protocol):
    name: str

    def score(
        self,
        previous_text: str,
        next_text: str,
    ) -> float:
        ...


class BatchedTransitionScorer(
    TransitionScorer,
    Protocol,
):
    def score_many(
        self,
        pairs: Sequence[TransitionPair],
    ) -> list[float]:
        ...


@dataclass(frozen=True, slots=True)
class TransitionMatrix:
    """
    scores[i][j] is the model score for:

        chunk_i -> chunk_j
    """

    scores: tuple[
        tuple[float, ...],
        ...,
    ]

    @property
    def size(self) -> int:
        return len(self.scores)

    def get(
        self,
        previous_chunk: int,
        next_chunk: int,
    ) -> float:
        return self.scores[
            previous_chunk
        ][next_chunk]


def build_transition_matrix(
    transcript: DialogueTranscript,
    scorer: TransitionScorer,
) -> TransitionMatrix:
    """
    Build the full directed transition matrix.

    If the scorer exposes score_many(), all directed pairs are
    evaluated in batches. Otherwise we fall back to score().
    """
    n = len(transcript.chunks)

    matrix: list[list[float]] = [
        [float("-inf")] * n
        for _ in range(n)
    ]

    pairs: list[TransitionPair] = []
    locations: list[tuple[int, int]] = []

    for i in range(n):
        previous_text = transcript.text_for(i)

        for j in range(n):
            if i == j:
                continue

            next_text = transcript.text_for(j)

            pairs.append(
                (
                    previous_text,
                    next_text,
                )
            )

            locations.append(
                (i, j)
            )

    score_many = getattr(
        scorer,
        "score_many",
        None,
    )

    if callable(score_many):
        values = score_many(
            pairs
        )
    else:
        values = [
            scorer.score(
                previous_text,
                next_text,
            )
            for (
                previous_text,
                next_text,
            )
            in pairs
        ]

    if len(values) != len(pairs):
        raise RuntimeError(
            "Transition scorer returned "
            "an unexpected number of scores"
        )

    for (
        (i, j),
        value,
    ) in zip(
        locations,
        values,
        strict=True,
    ):
        matrix[i][j] = float(
            value
        )

    return TransitionMatrix(
        scores=tuple(
            tuple(row)
            for row in matrix
        )
    )