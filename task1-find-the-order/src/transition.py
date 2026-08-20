from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.transcription import DialogueTranscript


class TransitionScorer(Protocol):
    name: str

    def score(
        self,
        previous_text: str,
        next_text: str,
    ) -> float:
        """
        Higher score means it is more plausible that
        `next_text` follows `previous_text`.
        """
        ...


@dataclass(frozen=True, slots=True)
class TransitionMatrix:
    """
    scores[i][j] represents how plausible it is that
    chunk j directly follows chunk i.
    """

    scores: tuple[tuple[float, ...], ...]

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
    n = len(transcript.chunks)

    matrix: list[list[float]] = [
        [float("-inf")] * n
        for _ in range(n)
    ]

    for i in range(n):
        previous_text = transcript.text_for(i)

        for j in range(n):
            if i == j:
                continue

            next_text = transcript.text_for(j)

            matrix[i][j] = scorer.score(
                previous_text,
                next_text,
            )

    return TransitionMatrix(
        scores=tuple(
            tuple(row)
            for row in matrix
        )
    )