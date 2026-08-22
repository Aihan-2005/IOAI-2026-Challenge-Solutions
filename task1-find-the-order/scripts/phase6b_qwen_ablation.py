from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
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
    DiagnosticsReport,
    evaluate_ordering_diagnostics,
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
from src.qwen_scorer import (
    QwenContextualLiftScorer,
    QwenScorerConfig,
)
from src.sqlite_score_cache import (
    SQLiteCachedTransitionScorer,
)
from src.transcript_io import (
    TranscriptStore,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Controlled heuristic vs Qwen "
            "ablation on identical dialogues."
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
        required=True,
    )

    parser.add_argument(
        "--transcript-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--score-cache",
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


def diagnostic_summary(
    report: DiagnosticsReport,
) -> dict[str, float | int]:
    return {
        "dialogue_count": (
            report.dialogue_count
        ),
        "decision_count": (
            report.decision_count
        ),
        "top1": (
            report.local_top1_accuracy
        ),
        "mrr": (
            report.local_mrr
        ),
        "margin": (
            report.mean_true_margin
        ),
        "global_score": (
            report.mean_pairwise_score
        ),
        "gold_above_prediction_rate": (
            report
            .gold_path_beats_prediction_rate
        ),
    }


def print_diagnostics(
    label: str,
    report: DiagnosticsReport,
) -> None:
    print(
        f"{label:<12} "
        f"Top-1="
        f"{100 * report.local_top1_accuracy:6.2f}%  "
        f"MRR="
        f"{report.local_mrr:.4f}  "
        f"Margin="
        f"{report.mean_true_margin:+.4f}  "
        f"Global="
        f"{report.mean_pairwise_score:6.2f}  "
        f"Gold>Beam="
        f"{100 * report.gold_path_beats_prediction_rate:6.2f}%"
    )


def main() -> None:
    args = parse_args()

    all_dialogues = load_dialogues(
        args.split_dir
    )

    dialogues = all_dialogues[
        : args.max_dialogues
    ]

    if not dialogues:
        raise RuntimeError(
            "No dialogues selected"
        )

    targets = load_rank_answers(
        args.answers
    )

    transcript_store = (
        TranscriptStore(
            args.transcript_dir
        )
    )

    header(
        "EXPERIMENT PROTOCOL"
    )

    print(
        "Dialogues:",
        len(dialogues),
    )

    print(
        "Dialogue IDs:",
        [
            dialogue.dialogue_id
            for dialogue in dialogues
        ],
    )

    print(
        "Beam width:",
        args.beam_width,
    )

    print(
        "Qwen model:",
        args.model_id,
    )

    header(
        "LOADING QWEN"
    )

    qwen_backend = (
        QwenContextualLiftScorer(
            QwenScorerConfig(
                model_id=args.model_id,
                batch_size=(
                    args.batch_size
                ),
            )
        )
    )

    cache_namespace = (
        "qwen-contextual-lift-v1"
        f"|{args.model_id}"
        "|prompt384"
        "|candidate128"
    )

    qwen_scorer = (
        SQLiteCachedTransitionScorer(
            scorer=qwen_backend,
            database_path=(
                args.score_cache
            ),
            namespace=(
                cache_namespace
            ),
        )
    )

    print(
        "Device:",
        qwen_backend.device,
    )

    print(
        "Existing cached transitions:",
        qwen_scorer.cache_size,
    )

    heuristic_scorer = (
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
                heuristic_scorer
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
            transition_scorer=(
                qwen_scorer
            ),
            beam_width=(
                args.beam_width
            ),
        )
    )

    header(
        "GLOBAL BENCHMARK"
    )

    baseline_benchmark = (
        evaluate_predictor(
            dialogues=dialogues,
            targets=targets,
            predictor=(
                baseline_predictor
            ),
        )
    )

    heuristic_benchmark = (
        evaluate_predictor(
            dialogues=dialogues,
            targets=targets,
            predictor=(
                heuristic_predictor
            ),
        )
    )

    qwen_benchmark = (
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
        f"{baseline_benchmark.mean_score:.2f}",
    )

    print(
        "Whisper + heuristic:",
        f"{heuristic_benchmark.mean_score:.2f}",
    )

    print(
        "Whisper + Qwen:",
        f"{qwen_benchmark.mean_score:.2f}",
    )

    print()

    print(
        "Heuristic vs baseline:",
        (
            f"{heuristic_benchmark.mean_score - baseline_benchmark.mean_score:+.2f}"
        ),
    )

    print(
        "Qwen vs baseline:",
        (
            f"{qwen_benchmark.mean_score - baseline_benchmark.mean_score:+.2f}"
        ),
    )

    print(
        "Qwen vs heuristic:",
        (
            f"{qwen_benchmark.mean_score - heuristic_benchmark.mean_score:+.2f}"
        ),
    )

    header(
        "LOCAL + GLOBAL DIAGNOSTICS"
    )

    heuristic_diagnostics = (
        evaluate_ordering_diagnostics(
            dialogues=dialogues,
            targets=targets,
            transcript_store=(
                transcript_store
            ),
            transition_scorer=(
                heuristic_scorer
            ),
            predictor=(
                heuristic_predictor
            ),
        )
    )

    qwen_diagnostics = (
        evaluate_ordering_diagnostics(
            dialogues=dialogues,
            targets=targets,
            transcript_store=(
                transcript_store
            ),
            transition_scorer=(
                qwen_scorer
            ),
            predictor=(
                qwen_predictor
            ),
        )
    )

    print_diagnostics(
        "Heuristic",
        heuristic_diagnostics,
    )

    print_diagnostics(
        "Qwen",
        qwen_diagnostics,
    )

    header(
        "CACHE"
    )

    print(
        "Persistent Qwen transitions:",
        qwen_scorer.cache_size,
    )

    print(
        "Database:",
        args.score_cache,
    )

    payload = {
        "created_at_utc": (
            datetime.now(UTC)
            .isoformat()
        ),
        "experiment": (
            "phase6b-fair-qwen-ablation"
        ),
        "dialogue_ids": [
            dialogue.dialogue_id
            for dialogue in dialogues
        ],
        "beam_width": (
            args.beam_width
        ),
        "model_id": (
            args.model_id
        ),
        "global_scores": {
            "baseline": (
                baseline_benchmark
                .mean_score
            ),
            "heuristic": (
                heuristic_benchmark
                .mean_score
            ),
            "qwen": (
                qwen_benchmark
                .mean_score
            ),
        },
        "diagnostics": {
            "heuristic": (
                diagnostic_summary(
                    heuristic_diagnostics
                )
            ),
            "qwen": (
                diagnostic_summary(
                    qwen_diagnostics
                )
            ),
        },
        "qwen_cache_entries": (
            qwen_scorer.cache_size
        ),
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = (
        args.output.with_suffix(
            args.output.suffix
            + ".tmp"
        )
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    temporary.replace(
        args.output
    )

    header(
        "RESULT"
    )

    print(
        "Report:",
        args.output,
    )


if __name__ == "__main__":
    main()