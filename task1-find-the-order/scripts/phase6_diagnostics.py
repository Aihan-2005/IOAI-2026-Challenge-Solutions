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


from src.diagnostic_io import (
    write_diagnostics_report,
)
from src.diagnostics import (
    evaluate_ordering_diagnostics,
)
from src.dataset_io import (
    load_dialogues,
)
from src.evaluation_io import (
    load_rank_answers,
)
from src.failure_analysis import (
    format_dialogue_failure,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose Task 1 ordering "
            "scorer and search behavior."
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
        "--show-failures",
        type=int,
        default=3,
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

    store = TranscriptStore(
        args.transcript_dir
    )

    scorer = (
        HeuristicDialogueTransitionScorer()
    )

    predictor = (
        TranscriptOrderingPredictor(
            transcript_store=store,
            transition_scorer=scorer,
            beam_width=(
                args.beam_width
            ),
        )
    )

    header(
        "PHASE 6 — SCORER / SEARCH DIAGNOSTICS"
    )

    report = (
        evaluate_ordering_diagnostics(
            dialogues=dialogues,
            targets=targets,
            transcript_store=store,
            transition_scorer=scorer,
            predictor=predictor,
        )
    )

    print(
        "Scorer:",
        report.scorer_name,
    )

    print(
        "Predictor:",
        report.predictor_name,
    )

    print(
        "Dialogues:",
        report.dialogue_count,
    )

    print(
        "Non-trivial decisions:",
        report.decision_count,
    )

    print()

    print(
        "Local next-turn Top-1:",
        f"{100 * report.local_top1_accuracy:.2f}%",
    )

    print(
        "Local next-turn MRR:",
        f"{report.local_mrr:.4f}",
    )

    print(
        "Mean true-vs-best-wrong margin:",
        f"{report.mean_true_margin:+.4f}",
    )

    print()

    print(
        "Global pairwise score:",
        f"{report.mean_pairwise_score:.2f}",
    )

    print(
        "Gold path scored above "
        "beam prediction:",
        (
            f"{100 * report.gold_path_beats_prediction_rate:.2f}%"
        ),
    )

    output = (
        write_diagnostics_report(
            report=report,
            output_path=args.output,
        )
    )

    header(
        "WORST GLOBAL CASES"
    )

    worst = sorted(
        report.dialogues,
        key=lambda item: (
            item.pairwise_score,
            -item.n_chunks,
            int(item.dialogue_id),
        ),
    )[
        : args.show_failures
    ]

    dialogue_lookup = {
        dialogue.dialogue_id:
        dialogue
        for dialogue in dialogues
    }

    for diagnostic in worst:
        dialogue = dialogue_lookup[
            diagnostic.dialogue_id
        ]

        target_rank = targets[
            diagnostic.dialogue_id
        ]

        predicted_rank = (
            predictor.predict(
                dialogue
            )
        )

        print()
        print(
            f"Score: "
            f"{100 * diagnostic.pairwise_score:.2f}"
        )

        print(
            f"Gold path score: "
            f"{diagnostic.gold_path_score:.4f}"
        )

        print(
            f"Predicted path score: "
            f"{diagnostic.predicted_path_score:.4f}"
        )

        print(
            "Gold > predicted objective:",
            diagnostic.gold_path_beats_prediction,
        )

        print()

        print(
            format_dialogue_failure(
                dialogue=dialogue,
                target_rank=target_rank,
                predicted_rank=(
                    predicted_rank
                ),
                transcript_store=store,
            )
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