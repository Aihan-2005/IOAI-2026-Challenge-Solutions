from __future__ import annotations

import sys
import tempfile
from pathlib import Path


TASK_ROOT = (
    Path(__file__).resolve().parents[1]
)

sys.path.insert(
    0,
    str(TASK_ROOT),
)


from src.domain import Dialogue
from src.global_ordering import (
    beam_search_order,
    greedy_order,
    path_score,
)
from src.heuristic_scorer import (
    HeuristicDialogueTransitionScorer,
)
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
from src.transition import (
    TransitionMatrix,
)


def header(
    title: str,
) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def audit_search() -> None:
    header(
        "1. GLOBAL ORDERING SEARCH"
    )

    matrix = TransitionMatrix(
        scores=(
            (
                float("-inf"),
                0.0,
                0.0,
                0.0,
                0.0,
            ),
            (
                0.0,
                float("-inf"),
                10.0,
                9.0,
                0.0,
            ),
            (
                0.0,
                0.0,
                float("-inf"),
                0.0,
                0.0,
            ),
            (
                0.0,
                0.0,
                0.0,
                float("-inf"),
                9.0,
            ),
            (
                0.0,
                0.0,
                9.0,
                0.0,
                float("-inf"),
            ),
        )
    )

    greedy = greedy_order(
        transition_matrix=matrix,
        prefix=(0, 1),
    )

    beam = beam_search_order(
        transition_matrix=matrix,
        prefix=(0, 1),
        beam_width=8,
    )

    greedy_score = path_score(
        greedy,
        matrix,
    )

    beam_score = path_score(
        beam,
        matrix,
    )

    print(
        f"Greedy: {greedy} "
        f"score={greedy_score}"
    )

    print(
        f"Beam:   {beam} "
        f"score={beam_score}"
    )

    assert beam_score > greedy_score

    print(
        "[OK] beam search escapes "
        "a greedy local optimum"
    )


def audit_heuristic_scorer() -> None:
    header(
        "2. LIGHTWEIGHT TRANSITION SCORER"
    )

    scorer = (
        HeuristicDialogueTransitionScorer()
    )

    question = (
        "Are you coming to the meeting?"
    )

    answer = (
        "No, I have a dentist appointment."
    )

    unrelated = (
        "I bought a new table yesterday."
    )

    answer_score = scorer.score(
        question,
        answer,
    )

    unrelated_score = scorer.score(
        question,
        unrelated,
    )

    print(
        f"Question -> Answer: "
        f"{answer_score:.3f}"
    )

    print(
        f"Question -> Unrelated: "
        f"{unrelated_score:.3f}"
    )

    assert (
        answer_score
        >
        unrelated_score
    )

    print(
        "[OK] dialogue heuristic "
        "detects stronger continuity"
    )


class StaticScorer:
    name = "audit-static"

    def score(
        self,
        previous_text: str,
        next_text: str,
    ) -> float:
        transitions = {
            (
                "second",
                "third",
            ): 10.0,
            (
                "third",
                "last",
            ): 10.0,
        }

        return transitions.get(
            (
                previous_text,
                next_text,
            ),
            0.0,
        )


def audit_gpu_mac_boundary(
    temporary_root: Path,
) -> None:
    header(
        "3. GPU -> TRANSCRIPT -> MAC BOUNDARY"
    )

    store = TranscriptStore(
        temporary_root
        / "transcripts"
    )

    transcript = DialogueTranscript(
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

    path = store.save(
        transcript
    )

    print(
        f"[OK] transcript artifact: "
        f"{path.name}"
    )

    loaded = store.load(
        "0"
    )

    assert loaded == transcript

    print(
        "[OK] transcript round-trip"
    )

    dialogue = Dialogue(
        dialogue_id="0",
        directory=(
            temporary_root / "0"
        ),
        chunk_paths=tuple(
            temporary_root
            / "0"
            / f"chunk_{index}.wav"
            for index in range(4)
        ),
        prefix=(1, 2),
    )

    predictor = (
        TranscriptOrderingPredictor(
            transcript_store=store,
            transition_scorer=(
                StaticScorer()
            ),
            beam_width=8,
        )
    )

    rank = predictor.predict(
        dialogue
    )

    assert rank == [
        3,
        0,
        1,
        2,
    ]

    print(
        f"[OK] predicted rank: {rank}"
    )


def main() -> None:
    header(
        "PHASE 4B ORDERING ENGINE AUDIT"
    )

    audit_search()

    audit_heuristic_scorer()

    with tempfile.TemporaryDirectory() as tmp:
        audit_gpu_mac_boundary(
            Path(tmp)
        )

    header(
        "PHASE 4B RESULT"
    )

    print(
        "All Phase 4B checks passed."
    )

    print()
    print("Validated:")
    print("- greedy ordering")
    print("- beam-search ordering")
    print("- path objective")
    print("- lightweight dialogue scoring")
    print("- persistent transcript artifacts")
    print("- transcript-based predictor")
    print("- GPU/Mac separation")
    print()
    print(
        "The local ordering engine is "
        "ready for real ASR integration."
    )


if __name__ == "__main__":
    main()