from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.domain import Dialogue
from src.global_ordering import (
    beam_search_order,
)
from src.order_utils import (
    order_to_rank,
)
from src.transcript_io import (
    TranscriptStore,
)
from src.transition import (
    TransitionScorer,
    build_transition_matrix,
)


class RankingPredictor(Protocol):
    """
    Interface implemented by every Task 1 solution.
    """

    name: str

    def predict(
        self,
        dialogue: Dialogue,
    ) -> list[int]:
        ...


class PrefixIndexBaseline:
    """
    Reimplementation of the official deterministic baseline.
    """

    name = "prefix-index-baseline"

    def predict(
        self,
        dialogue: Dialogue,
    ) -> list[int]:
        first, second = (
            dialogue.prefix
        )

        chronological_order = [
            first,
            second,
            *(
                index
                for index in range(
                    dialogue.n_chunks
                )
                if index
                not in (
                    first,
                    second,
                )
            ),
        ]

        return order_to_rank(
            chronological_order
        )


@dataclass(slots=True)
class TranscriptOrderingPredictor:
    """
    Text-based ordering predictor.

    The expensive audio -> text step is intentionally
    outside this class.

    This predictor consumes cached transcript artifacts.
    """

    transcript_store: TranscriptStore
    transition_scorer: TransitionScorer
    beam_width: int = 32

    @property
    def name(self) -> str:
        return (
            "transcript-beam-"
            f"{self.transition_scorer.name}"
        )

    def predict(
        self,
        dialogue: Dialogue,
    ) -> list[int]:
        transcript = (
            self.transcript_store.load(
                dialogue.dialogue_id
            )
        )

        if (
            transcript.dialogue_id
            != dialogue.dialogue_id
        ):
            raise ValueError(
                "transcript dialogue ID mismatch"
            )

        if (
            len(transcript.chunks)
            != dialogue.n_chunks
        ):
            raise ValueError(
                f"Dialogue {dialogue.dialogue_id}: "
                "transcript chunk count does not "
                "match audio chunk count"
            )

        matrix = (
            build_transition_matrix(
                transcript=transcript,
                scorer=(
                    self.transition_scorer
                ),
            )
        )

        chronological_order = (
            beam_search_order(
                transition_matrix=matrix,
                prefix=dialogue.prefix,
                beam_width=self.beam_width,
            )
        )

        return order_to_rank(
            chronological_order
        )