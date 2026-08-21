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


from src.benchmark import (
    evaluate_predictor,
)
from src.dataset_io import (
    load_dialogues,
)
from src.diagnostics import (
    evaluate_ordering_diagnostics,
)
from src.evaluation_io import (
    load_rank_answers,
)
from src.heuristic_scorer import (
    HeuristicDialogueTransitionScorer,
)
from src.memoized_scorer import (
    MemoizedTransitionScorer,
)
from src.predictors import (
    PrefixIndexBaseline,
    TranscriptOrderingPredictor,
)
from src.qwen_scorer import (
    QwenContextualLiftScorer,
    QwenScorerConfig,
)
from src.transcript_io import (
    TranscriptStore,
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
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--model-id",
        default=(
            "Qwen/Qwen2.5-0.5B"
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
    )[
        : args.max_dialogues
    ]

    targets = load_rank_answers(
        args.answers
    )

    transcript_store = (
        TranscriptStore(
            args.transcript_dir
        )
    )

    header(
        "LOADING QWEN"
    )

    qwen_base = (
        QwenContextualLiftScorer(
            QwenScorerConfig(
                model_id=(
                    args.model_id
                ),
                batch_size=(
                    args.batch_size
                ),
            )
        )
    )

    qwen = (
        MemoizedTransitionScorer(
            qwen_base
        )
    )

    print(
        "Device:",
        qwen_base.device,
    )

    heuristic = (
        HeuristicDialogueTransitionScorer()
    )

    baseline_predictor = (
        PrefixIndexBaseline()
    )

    heuristic_predictor = (
        TranscriptOrderingPredictor(
            transcript_store=(
                transcript_store
            ),
            transition_scorer=(
                heuristic
            ),
            beam_width=(
                args.beam_width
            ),
        )
    )

    qwen_predictor = (
        TranscriptOrderingPredictor(
            transcript_store=(
                transcript_store
            ),
            transition_scorer=qwen,
            beam_width=(
                args.beam_width
            ),
        )
    )

    header(
        "CONTROLLED ABLATION"
    )

    baseline = evaluate_predictor(
        dialogues=dialogues,
        targets=targets,
        predictor=(
            baseline_predictor
        ),
    )

    heuristic_result = (
        evaluate_predictor(
            dialogues=dialogues,
            targets=targets,
            predictor=(
                heuristic_predictor
            ),
        )
    )

    qwen_result = (
        evaluate_predictor(
            dialogues=dialogues,
            targets=targets,
            predictor=(
                qwen_predictor
            ),
        )
    )

    print(
        "Official baseline:",
        f"{baseline.mean_score:.2f}",
    )

    print(
        "Whisper + heuristic:",
        f"{heuristic_result.mean_score:.2f}",
    )

    print(
        "Whisper + Qwen:",
        f"{qwen_result.mean_score:.2f}",
    )

    print()

    print(
        "Qwen vs heuristic:",
        (
            f"{qwen_result.mean_score - heuristic_result.mean_score:+.2f}"
        ),
    )

    print(
        "Qwen cache entries:",
        qwen.cache_size,
    )

    header(
        "QWEN DIAGNOSTICS"
    )

    diagnostics = (
        evaluate_ordering_diagnostics(
            dialogues=dialogues,
            targets=targets,
            transcript_store=(
                transcript_store
            ),
            transition_scorer=qwen,
            predictor=(
                qwen_predictor
            ),
        )
    )

    print(
        "Local next-turn Top-1:",
        (
            f"{100 * diagnostics.local_top1_accuracy:.2f}%"
        ),
    )

    print(
        "Local next-turn MRR:",
        f"{diagnostics.local_mrr:.4f}",
    )

    print(
        "Mean true-vs-best-wrong margin:",
        (
            f"{diagnostics.mean_true_margin:+.4f}"
        ),
    )

    print(
        "Global pairwise score:",
        (
            f"{diagnostics.mean_pairwise_score:.2f}"
        ),
    )

    print(
        "Gold path scored above beam prediction:",
        (
            f"{100 * diagnostics.gold_path_beats_prediction_rate:.2f}%"
        ),
    )


if __name__ == "__main__":
    main()