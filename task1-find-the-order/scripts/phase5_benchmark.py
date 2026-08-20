from __future__ import annotations

import argparse
import sys
from pathlib import Path


TASK_ROOT = (
    Path(__file__).resolve().parents[1]
)

sys.path.insert(
    0,
    str(TASK_ROOT),
)


from src.analysis import (
    compare_predictors,
    worst_dialogues,
)
from src.asr_whisper import (
    WhisperConfig,
    WhisperTranscriber,
)
from src.benchmark import (
    build_report,
    evaluate_predictor,
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
from src.predictors import (
    PrefixIndexBaseline,
    TranscriptOrderingPredictor,
)
from src.report_io import (
    write_benchmark_report,
)
from src.transcript_io import (
    TranscriptStore,
)
from src.transcription_runner import (
    populate_transcript_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--answers",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--transcript-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--max-dialogues",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--beam-width",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--model-id",
        default=(
            "openai/whisper-small"
        ),
    )

    return parser.parse_args()


def header(
    title: str,
) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    args = parse_args()

    dialogues = load_dialogues(
        args.split_dir
    )

    selected = dialogues[
        : args.max_dialogues
    ]

    targets = load_rank_answers(
        args.answers
    )

    header(
        "TRANSCRIPT CACHE"
    )

    store = TranscriptStore(
        args.transcript_dir
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

    stats = populate_transcript_cache(
        dialogues=selected,
        transcriber=transcriber,
        store=store,
    )

    print()
    print(
        "Transcribed:",
        stats.transcribed_dialogues,
    )

    print(
        "Cached:",
        stats.cached_dialogues,
    )

    print(
        "Elapsed:",
        f"{stats.elapsed_seconds:.1f}s",
    )

    header(
        "BENCHMARK"
    )

    baseline = evaluate_predictor(
        dialogues=selected,
        targets=targets,
        predictor=(
            PrefixIndexBaseline()
        ),
    )

    candidate = evaluate_predictor(
        dialogues=selected,
        targets=targets,
        predictor=(
            TranscriptOrderingPredictor(
                transcript_store=store,
                transition_scorer=(
                    HeuristicDialogueTransitionScorer()
                ),
                beam_width=(
                    args.beam_width
                ),
            )
        ),
    )

    print(
        "Official baseline:",
        f"{baseline.mean_score:.2f}",
    )

    print(
        "Whisper + heuristic:",
        f"{candidate.mean_score:.2f}",
    )

    header(
        "WORST CANDIDATE DIALOGUES"
    )

    for result in worst_dialogues(
        candidate,
        limit=5,
    ):
        print(
            f"id={result.dialogue_id:<4} "
            f"chunks={result.n_chunks:<2} "
            f"score={100 * result.score:.2f}"
        )

    header(
        "BIGGEST DELTAS"
    )

    deltas = compare_predictors(
        baseline=baseline,
        candidate=candidate,
    )

    for dialogue_id, delta in deltas:
        print(
            f"id={dialogue_id:<4} "
            f"delta={100 * delta:+.2f}"
        )

    report = build_report(
        dataset_name=(
            args.split_dir.name
        ),
        benchmarks=(
            baseline,
            candidate,
        ),
    )

    output = write_benchmark_report(
        report=report,
        output_path=args.output,
    )

    header(
        "REPORT"
    )

    print(
        "Saved:",
        output,
    )


if __name__ == "__main__":
    main()