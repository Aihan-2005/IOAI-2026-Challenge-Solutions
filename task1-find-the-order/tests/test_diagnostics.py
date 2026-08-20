from pathlib import Path

from src.diagnostics import (
    evaluate_ordering_diagnostics,
)
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


class PerfectTransitionScorer:
    name = "perfect-test-scorer"

    def score(
        self,
        previous_text: str,
        next_text: str,
    ) -> float:
        scores = {
            (
                "first",
                "second",
            ): 10.0,
            (
                "second",
                "third",
            ): 10.0,
            (
                "second",
                "fourth",
            ): 1.0,
            (
                "third",
                "fourth",
            ): 10.0,
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
        prefix=(0, 1),
    )


def _store(
    tmp_path,
) -> TranscriptStore:
    store = TranscriptStore(
        tmp_path
        / "transcripts"
    )

    store.save(
        DialogueTranscript(
            dialogue_id="0",
            chunks=(
                ChunkTranscript(
                    0,
                    "first",
                ),
                ChunkTranscript(
                    1,
                    "second",
                ),
                ChunkTranscript(
                    2,
                    "third",
                ),
                ChunkTranscript(
                    3,
                    "fourth",
                ),
            ),
        )
    )

    return store


def test_diagnostics_identify_perfect_local_choice(
    tmp_path,
):
    scorer = (
        PerfectTransitionScorer()
    )

    store = _store(
        tmp_path
    )

    predictor = (
        TranscriptOrderingPredictor(
            transcript_store=store,
            transition_scorer=scorer,
            beam_width=8,
        )
    )

    report = (
        evaluate_ordering_diagnostics(
            dialogues=(
                _dialogue(),
            ),
            targets={
                "0": [
                    0,
                    1,
                    2,
                    3,
                ],
            },
            transcript_store=store,
            transition_scorer=scorer,
            predictor=predictor,
        )
    )

    assert (
        report.local_top1_accuracy
        == 1.0
    )

    assert (
        report.local_mrr
        == 1.0
    )

    assert (
        report.mean_pairwise_score
        == 100.0
    )


class MisleadingTransitionScorer:
    name = "misleading-test-scorer"

    def score(
        self,
        previous_text: str,
        next_text: str,
    ) -> float:
        scores = {
            (
                "second",
                "fourth",
            ): 10.0,
            (
                "second",
                "third",
            ): 1.0,
            (
                "fourth",
                "third",
            ): 10.0,
        }

        return scores.get(
            (
                previous_text,
                next_text,
            ),
            0.0,
        )


def test_diagnostics_detect_bad_scorer(
    tmp_path,
):
    scorer = (
        MisleadingTransitionScorer()
    )

    store = _store(
        tmp_path
    )

    predictor = (
        TranscriptOrderingPredictor(
            transcript_store=store,
            transition_scorer=scorer,
            beam_width=8,
        )
    )

    report = (
        evaluate_ordering_diagnostics(
            dialogues=(
                _dialogue(),
            ),
            targets={
                "0": [
                    0,
                    1,
                    2,
                    3,
                ],
            },
            transcript_store=store,
            transition_scorer=scorer,
            predictor=predictor,
        )
    )

    assert (
        report.local_top1_accuracy
        == 0.0
    )

    assert (
        report.mean_pairwise_score
        < 100.0
    )