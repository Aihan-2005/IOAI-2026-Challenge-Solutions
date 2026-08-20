from src.transcription import (
    ChunkTranscript,
    DialogueTranscript,
)
from src.transition import (
    build_transition_matrix,
)


class ToyScorer:
    name = "toy-scorer"

    def score(
        self,
        previous_text: str,
        next_text: str,
    ) -> float:
        known_transitions = {
            (
                "Are you coming to the meeting?",
                "No, I have a dentist appointment.",
            ): 10.0,
            (
                "No, I have a dentist appointment.",
                "No problem, I will send you the notes.",
            ): 9.0,
        }

        return known_transitions.get(
            (previous_text, next_text),
            0.0,
        )


def test_transition_matrix():
    transcript = DialogueTranscript(
        dialogue_id="0",
        chunks=(
            ChunkTranscript(
                0,
                "No problem, I will send you the notes.",
            ),
            ChunkTranscript(
                1,
                "Are you coming to the meeting?",
            ),
            ChunkTranscript(
                2,
                "No, I have a dentist appointment.",
            ),
        ),
    )

    matrix = build_transition_matrix(
        transcript,
        ToyScorer(),
    )

    assert matrix.size == 3

    assert matrix.get(1, 2) == 10.0
    assert matrix.get(2, 0) == 9.0

    assert matrix.get(0, 0) == float("-inf")