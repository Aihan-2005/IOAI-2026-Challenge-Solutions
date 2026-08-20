from pathlib import Path

from src.domain import Dialogue
from src.predictors import (
    TranscriptOrderingPredictor,
)
from src.transcript_io import (
    TranscriptStore,
)
from src.transcription import (
    ChunkTranscript,
    DialogueTranscript,
)


class StaticTransitionScorer:
    name = "static-test-scorer"

    def score(
        self,
        previous_text: str,
        next_text: str,
    ) -> float:
        scores = {
            (
                "second",
                "third",
            ): 10.0,
            (
                "third",
                "last",
            ): 10.0,
            (
                "second",
                "last",
            ): 1.0,
            (
                "last",
                "third",
            ): 0.0,
        }

        return scores.get(
            (
                previous_text,
                next_text,
            ),
            0.0,
        )


def _dialogue() -> Dialogue:
    return Dialogue(
        dialogue_id="0",
        directory=Path(
            "/synthetic/0"
        ),
        chunk_paths=tuple(
            Path(
                f"/synthetic/0/"
                f"chunk_{index}.wav"
            )
            for index in range(4)
        ),
        prefix=(1, 2),
    )


def test_transcript_predictor_orders_dialogue(
    tmp_path,
):
    store = TranscriptStore(
        tmp_path / "transcripts"
    )

    store.save(
        DialogueTranscript(
            dialogue_id="0",
            chunks=(
                ChunkTranscript(
                    0,
                    "last",
                ),
                ChunkTranscript(
                    1,
                    "first",
                ),
                ChunkTranscript(
                    2,
                    "second",
                ),
                ChunkTranscript(
                    3,
                    "third",
                ),
            ),
        )
    )

    predictor = (
        TranscriptOrderingPredictor(
            transcript_store=store,
            transition_scorer=(
                StaticTransitionScorer()
            ),
            beam_width=8,
        )
    )

    rank = predictor.predict(
        _dialogue()
    )

    # Chronological order:
    #
    # 1 -> 2 -> 3 -> 0
    #
    # Rank:
    #
    # chunk 0 -> 3
    # chunk 1 -> 0
    # chunk 2 -> 1
    # chunk 3 -> 2

    assert rank == [
        3,
        0,
        1,
        2,
    ]


def test_predictor_name_contains_scorer(
    tmp_path,
):
    predictor = (
        TranscriptOrderingPredictor(
            transcript_store=(
                TranscriptStore(
                    tmp_path
                )
            ),
            transition_scorer=(
                StaticTransitionScorer()
            ),
        )
    )

    assert (
        predictor.name
        ==
        "transcript-beam-static-test-scorer"
    )