from src.transcription import (
    ChunkTranscript,
    DialogueTranscript,
)
from src.transition import (
    build_transition_matrix,
)


class FakeBatchedScorer:
    name = "fake-batched"

    def __init__(
        self,
    ) -> None:
        self.batch_calls = 0

    def score(
        self,
        previous_text: str,
        next_text: str,
    ) -> float:
        raise AssertionError(
            "score() should not be used"
        )

    def score_many(
        self,
        pairs,
    ):
        self.batch_calls += 1

        return [
            float(index)
            for index in range(
                len(pairs)
            )
        ]


def test_transition_matrix_uses_batch_api():
    transcript = DialogueTranscript(
        dialogue_id="0",
        chunks=(
            ChunkTranscript(
                0,
                "a",
            ),
            ChunkTranscript(
                1,
                "b",
            ),
            ChunkTranscript(
                2,
                "c",
            ),
        ),
    )

    scorer = FakeBatchedScorer()

    matrix = build_transition_matrix(
        transcript=transcript,
        scorer=scorer,
    )

    assert scorer.batch_calls == 1

    assert matrix.size == 3

    assert (
        matrix.get(0, 0)
        == float("-inf")
    )


def test_all_directed_pairs_are_scored():
    transcript = DialogueTranscript(
        dialogue_id="0",
        chunks=(
            ChunkTranscript(0, "a"),
            ChunkTranscript(1, "b"),
            ChunkTranscript(2, "c"),
            ChunkTranscript(3, "d"),
        ),
    )

    scorer = FakeBatchedScorer()

    build_transition_matrix(
        transcript=transcript,
        scorer=scorer,
    )

    assert scorer.batch_calls == 1