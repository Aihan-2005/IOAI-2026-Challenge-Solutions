from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path


TASK_ROOT = (
    Path(__file__).resolve().parents[1]
)

sys.path.insert(
    0,
    str(TASK_ROOT),
)


from src.asr_whisper import (
    WhisperConfig,
    WhisperTranscriber,
)
from src.dataset_io import (
    load_dialogues,
)
from src.evaluation_io import (
    load_rank_answers,
)
from src.heuristic_scorer import (
    HeuristicDialogueTransitionScorer,
)
from src.order_utils import (
    pairwise_score,
)
from src.predictors import (
    PrefixIndexBaseline,
    TranscriptOrderingPredictor,
)
from src.transcript_io import (
    TranscriptStore,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "IOAI Task 1 real-data "
            "GPU smoke test"
        )
    )

    parser.add_argument(
        "--split-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--answers",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--transcript-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--max-dialogues",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--model-id",
        default=(
            "openai/whisper-small"
        ),
    )

    parser.add_argument(
        "--beam-width",
        type=int,
        default=32,
    )

    return parser.parse_args()


def print_header(
    title: str,
) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    args = parse_args()

    print_header(
        "PHASE 5 REAL DATA SMOKE TEST"
    )

    dialogues = load_dialogues(
        args.split_dir
    )

    selected = dialogues[
        : args.max_dialogues
    ]

    if not selected:
        raise RuntimeError(
            "No dialogues found"
        )

    print(
        f"Loaded dialogues: "
        f"{len(dialogues)}"
    )

    print(
        f"Selected dialogues: "
        f"{len(selected)}"
    )

    answers = None

    if args.answers is not None:
        answers = load_rank_answers(
            args.answers
        )

        print(
            f"Ground truth loaded: "
            f"{len(answers)} dialogues"
        )

    store = TranscriptStore(
        args.transcript_dir
    )

    missing = [
        dialogue
        for dialogue in selected
        if not store.path_for(
            dialogue.dialogue_id
        ).is_file()
    ]

    transcriber = None

    if missing:
        print_header(
            "LOADING WHISPER"
        )

        transcriber = (
            WhisperTranscriber(
                WhisperConfig(
                    model_id=(
                        args.model_id
                    )
                )
            )
        )

        print(
            "Device:",
            transcriber.device,
        )

        print(
            "DType:",
            transcriber.dtype,
        )

    for dialogue in selected:
        transcript_path = (
            store.path_for(
                dialogue.dialogue_id
            )
        )

        if transcript_path.is_file():
            print(
                f"[CACHE] dialogue "
                f"{dialogue.dialogue_id}"
            )

            continue

        if transcriber is None:
            raise RuntimeError(
                "Transcriber was not initialized"
            )

        print_header(
            "TRANSCRIBING DIALOGUE "
            f"{dialogue.dialogue_id}"
        )

        transcript = (
            transcriber.transcribe(
                dialogue
            )
        )

        store.save(
            transcript
        )

        for chunk in transcript.chunks:
            print(
                f"[{chunk.chunk_index:02d}] "
                f"{chunk.text}"
            )

    baseline = (
        PrefixIndexBaseline()
    )

    semantic_predictor = (
        TranscriptOrderingPredictor(
            transcript_store=store,
            transition_scorer=(
                HeuristicDialogueTransitionScorer()
            ),
            beam_width=args.beam_width,
        )
    )

    baseline_scores: list[
        float
    ] = []

    semantic_scores: list[
        float
    ] = []

    print_header(
        "PREDICTIONS"
    )

    for dialogue in selected:
        baseline_rank = (
            baseline.predict(
                dialogue
            )
        )

        semantic_rank = (
            semantic_predictor.predict(
                dialogue
            )
        )

        print()
        print(
            "Dialogue:",
            dialogue.dialogue_id,
        )

        print(
            "Prefix:",
            dialogue.prefix,
        )

        print(
            "Baseline:",
            baseline_rank,
        )

        print(
            "Whisper + heuristic:",
            semantic_rank,
        )

        if (
            answers is not None
            and dialogue.dialogue_id
            in answers
        ):
            truth = answers[
                dialogue.dialogue_id
            ]

            baseline_score = (
                pairwise_score(
                    baseline_rank,
                    truth,
                )
            )

            semantic_score = (
                pairwise_score(
                    semantic_rank,
                    truth,
                )
            )

            baseline_scores.append(
                baseline_score
            )

            semantic_scores.append(
                semantic_score
            )

            print(
                "Truth:",
                truth,
            )

            print(
                "Baseline score:",
                f"{baseline_score:.4f}",
            )

            print(
                "Semantic score:",
                f"{semantic_score:.4f}",
            )

    if baseline_scores:
        print_header(
            "SMOKE TEST SCORE SUMMARY"
        )

        print(
            "Official baseline:",
            f"{100 * statistics.mean(baseline_scores):.2f}",
        )

        print(
            "Whisper + heuristic:",
            f"{100 * statistics.mean(semantic_scores):.2f}",
        )

    print_header(
        "PHASE 5 SMOKE TEST COMPLETE"
    )

    print(
        "Real audio -> Whisper -> transcript "
        "-> ordering pipeline executed successfully."
    )


if __name__ == "__main__":
    main()