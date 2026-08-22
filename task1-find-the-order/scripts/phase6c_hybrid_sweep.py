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
from src.dialogue_selection import (
    deterministic_dialogue_sample,
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
from src.hybrid_scorer import (
    HybridTransitionScorer,
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


WEIGHTS = (
    0.20,
    0.40,
    0.60,
    0.80,
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
        "--count",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--seed",
        default="phase6c-dev-v1",
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

    return parser.parse_args()


def evaluate(
    *,
    label: str,
    dialogues,
    targets,
    store,
    scorer,
    beam_width: int,
) -> dict:
    predictor = (
        TranscriptOrderingPredictor(
            transcript_store=store,
            transition_scorer=scorer,
            beam_width=beam_width,
        )
    )

    benchmark = evaluate_predictor(
        dialogues=dialogues,
        targets=targets,
        predictor=predictor,
    )

    diagnostics = (
        evaluate_ordering_diagnostics(
            dialogues=dialogues,
            targets=targets,
            transcript_store=store,
            transition_scorer=scorer,
            predictor=predictor,
        )
    )

    return {
        "label": label,
        "global_score": (
            benchmark.mean_score
        ),
        "top1": (
            diagnostics
            .local_top1_accuracy
        ),
        "mrr": (
            diagnostics.local_mrr
        ),
        "margin": (
            diagnostics
            .mean_true_margin
        ),
        "gold_above_beam": (
            diagnostics
            .gold_path_beats_prediction_rate
        ),
    }


def print_result(
    result: dict,
) -> None:
    print(
        f"{result['label']:<22} "
        f"Global={result['global_score']:6.2f}  "
        f"Top1={100 * result['top1']:6.2f}%  "
        f"MRR={result['mrr']:.4f}  "
        f"Margin={result['margin']:+.4f}  "
        f"Gold>Beam="
        f"{100 * result['gold_above_beam']:6.2f}%"
    )


def main() -> None:
    args = parse_args()

    all_dialogues = load_dialogues(
        args.split_dir
    )

    dialogues = (
        deterministic_dialogue_sample(
            all_dialogues,
            count=args.count,
            seed=args.seed,
        )
    )

    targets = load_rank_answers(
        args.answers
    )

    store = TranscriptStore(
        args.transcript_dir
    )

    missing = [
        dialogue.dialogue_id
        for dialogue in dialogues
        if not store.path_for(
            dialogue.dialogue_id
        ).is_file()
    ]

    if missing:
        raise RuntimeError(
            "Missing cached transcripts for: "
            + ", ".join(missing)
        )

    qwen_backend = (
        QwenContextualLiftScorer(
            QwenScorerConfig(
                batch_size=(
                    args.batch_size
                )
            )
        )
    )

    qwen = (
        SQLiteCachedTransitionScorer(
            scorer=qwen_backend,
            database_path=(
                args.score_cache
            ),
            namespace=(
                "qwen-contextual-lift-v1"
                "|train-dev"
            ),
        )
    )

    heuristic = (
        HeuristicDialogueTransitionScorer()
    )

    baseline = evaluate_predictor(
        dialogues=dialogues,
        targets=targets,
        predictor=(
            PrefixIndexBaseline()
        ),
    )

    print()
    print(
        "Baseline:",
        f"{baseline.mean_score:.2f}",
    )

    print()
    print(
        "=== RAW CONTROLS ==="
    )

    heuristic_result = evaluate(
        label="Heuristic",
        dialogues=dialogues,
        targets=targets,
        store=store,
        scorer=heuristic,
        beam_width=args.beam_width,
    )

    qwen_result = evaluate(
        label="Qwen",
        dialogues=dialogues,
        targets=targets,
        store=store,
        scorer=qwen,
        beam_width=args.beam_width,
    )

    print_result(
        heuristic_result
    )

    print_result(
        qwen_result
    )

    print()
    print(
        "=== HYBRID SWEEP ==="
    )

    hybrid_results = []

    for weight in WEIGHTS:
        scorer = (
            HybridTransitionScorer(
                heuristic_scorer=(
                    heuristic
                ),
                semantic_scorer=qwen,
                semantic_weight=(
                    weight
                ),
            )
        )

        result = evaluate(
            label=(
                f"Hybrid qwen={weight:.2f}"
            ),
            dialogues=dialogues,
            targets=targets,
            store=store,
            scorer=scorer,
            beam_width=(
                args.beam_width
            ),
        )

        hybrid_results.append(
            result
        )

        print_result(
            result
        )

    best = max(
        hybrid_results,
        key=lambda item: (
            item["global_score"],
            item["mrr"],
        ),
    )

    print()
    print(
        "=== BEST TRAIN-DEV HYBRID ==="
    )

    print_result(
        best
    )

    payload = {
        "created_at_utc": (
            datetime.now(UTC)
            .isoformat()
        ),
        "seed": args.seed,
        "dialogue_ids": [
            dialogue.dialogue_id
            for dialogue in dialogues
        ],
        "baseline_score": (
            baseline.mean_score
        ),
        "controls": {
            "heuristic": (
                heuristic_result
            ),
            "qwen": (
                qwen_result
            ),
        },
        "hybrid_results": (
            hybrid_results
        ),
        "best_hybrid": (
            best
        ),
        "qwen_cache_entries": (
            qwen.cache_size
        ),
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Report:",
        args.output,
    )


if __name__ == "__main__":
    main()