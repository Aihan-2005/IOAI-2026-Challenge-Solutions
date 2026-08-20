from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from src.domain import Dialogue
from src.global_ordering import path_score
from src.order_utils import (
    pairwise_score,
    rank_to_order,
)
from src.predictors import RankingPredictor
from src.transcript_io import TranscriptStore
from src.transition import (
    TransitionScorer,
    build_transition_matrix,
)


@dataclass(frozen=True, slots=True)
class TurnDecisionDiagnostic:
    dialogue_id: str

    current_chunk: int
    true_next_chunk: int

    candidate_count: int

    true_score: float
    best_wrong_score: float

    true_candidate_rank: int
    reciprocal_rank: float
    margin: float

    top1_correct: bool


@dataclass(frozen=True, slots=True)
class DialogueDiagnostic:
    dialogue_id: str

    n_chunks: int

    pairwise_score: float

    truth_order: tuple[int, ...]
    predicted_order: tuple[int, ...]

    gold_path_score: float
    predicted_path_score: float

    gold_path_beats_prediction: bool

    decisions: tuple[
        TurnDecisionDiagnostic,
        ...,
    ]


@dataclass(frozen=True, slots=True)
class DiagnosticsReport:
    scorer_name: str
    predictor_name: str

    dialogue_count: int
    decision_count: int

    local_top1_accuracy: float
    local_mrr: float
    mean_true_margin: float

    mean_pairwise_score: float

    gold_path_beats_prediction_rate: float

    dialogues: tuple[
        DialogueDiagnostic,
        ...,
    ]


def _diagnose_gold_decisions(
    *,
    dialogue: Dialogue,
    truth_order: list[int],
    transition_matrix,
) -> tuple[
    TurnDecisionDiagnostic,
    ...,
]:
    """
    Evaluate scorer quality while following the true path.

    At each step we ask:

        Among all still-unseen chunks,
        where does the real next chunk rank?

    Forced final choices are excluded because they provide
    no information about scorer quality.
    """
    diagnostics: list[
        TurnDecisionDiagnostic
    ] = []

    # position 0 -> 1 is given by prefix.json.
    #
    # Start from the known second chunk.
    for position in range(
        1,
        len(truth_order) - 1,
    ):
        current_chunk = truth_order[
            position
        ]

        remaining = truth_order[
            position + 1 :
        ]

        # If only one chunk remains,
        # there is no ranking decision.
        if len(remaining) < 2:
            continue

        true_next = remaining[0]

        scored_candidates = [
            (
                candidate,
                transition_matrix.get(
                    current_chunk,
                    candidate,
                ),
            )
            for candidate in remaining
        ]

        ranked_candidates = sorted(
            scored_candidates,
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )

        true_candidate_rank = next(
            rank
            for rank, (
                candidate,
                _,
            ) in enumerate(
                ranked_candidates,
                start=1,
            )
            if candidate == true_next
        )

        true_score = (
            transition_matrix.get(
                current_chunk,
                true_next,
            )
        )

        wrong_scores = [
            score
            for candidate, score
            in scored_candidates
            if candidate != true_next
        ]

        best_wrong_score = max(
            wrong_scores
        )

        diagnostics.append(
            TurnDecisionDiagnostic(
                dialogue_id=(
                    dialogue.dialogue_id
                ),
                current_chunk=(
                    current_chunk
                ),
                true_next_chunk=(
                    true_next
                ),
                candidate_count=(
                    len(remaining)
                ),
                true_score=true_score,
                best_wrong_score=(
                    best_wrong_score
                ),
                true_candidate_rank=(
                    true_candidate_rank
                ),
                reciprocal_rank=(
                    1.0
                    / true_candidate_rank
                ),
                margin=(
                    true_score
                    - best_wrong_score
                ),
                top1_correct=(
                    true_candidate_rank
                    == 1
                ),
            )
        )

    return tuple(
        diagnostics
    )


def evaluate_ordering_diagnostics(
    *,
    dialogues: tuple[
        Dialogue,
        ...,
    ],
    targets: dict[
        str,
        list[int],
    ],
    transcript_store: TranscriptStore,
    transition_scorer: TransitionScorer,
    predictor: RankingPredictor,
) -> DiagnosticsReport:
    """
    Measure both local scorer quality and global ordering quality.
    """
    dialogue_reports: list[
        DialogueDiagnostic
    ] = []

    all_decisions: list[
        TurnDecisionDiagnostic
    ] = []

    for dialogue in dialogues:
        dialogue_id = (
            dialogue.dialogue_id
        )

        if dialogue_id not in targets:
            raise KeyError(
                f"Missing target for dialogue "
                f"{dialogue_id}"
            )

        target_rank = targets[
            dialogue_id
        ]

        truth_order = rank_to_order(
            target_rank
        )

        transcript = (
            transcript_store.load(
                dialogue_id
            )
        )

        if (
            len(transcript.chunks)
            != dialogue.n_chunks
        ):
            raise ValueError(
                f"Dialogue {dialogue_id}: "
                "transcript/audio chunk count mismatch"
            )

        matrix = build_transition_matrix(
            transcript=transcript,
            scorer=transition_scorer,
        )

        predicted_rank = (
            predictor.predict(
                dialogue
            )
        )

        predicted_order = (
            rank_to_order(
                predicted_rank
            )
        )

        local_diagnostics = (
            _diagnose_gold_decisions(
                dialogue=dialogue,
                truth_order=truth_order,
                transition_matrix=matrix,
            )
        )

        all_decisions.extend(
            local_diagnostics
        )

        gold_score = path_score(
            truth_order,
            matrix,
        )

        predicted_score = path_score(
            predicted_order,
            matrix,
        )

        dialogue_reports.append(
            DialogueDiagnostic(
                dialogue_id=(
                    dialogue_id
                ),
                n_chunks=(
                    dialogue.n_chunks
                ),
                pairwise_score=(
                    pairwise_score(
                        predicted_rank,
                        target_rank,
                    )
                ),
                truth_order=tuple(
                    truth_order
                ),
                predicted_order=tuple(
                    predicted_order
                ),
                gold_path_score=(
                    gold_score
                ),
                predicted_path_score=(
                    predicted_score
                ),
                gold_path_beats_prediction=(
                    gold_score
                    > predicted_score
                ),
                decisions=(
                    local_diagnostics
                ),
            )
        )

    if not dialogue_reports:
        raise ValueError(
            "No dialogues provided"
        )

    if not all_decisions:
        raise ValueError(
            "No non-trivial ordering decisions found"
        )

    local_top1 = mean(
        1.0
        if decision.top1_correct
        else 0.0
        for decision in all_decisions
    )

    local_mrr = mean(
        decision.reciprocal_rank
        for decision in all_decisions
    )

    mean_margin = mean(
        decision.margin
        for decision in all_decisions
    )

    mean_pairwise = mean(
        report.pairwise_score
        for report in dialogue_reports
    )

    search_gap_rate = mean(
        1.0
        if report.gold_path_beats_prediction
        else 0.0
        for report in dialogue_reports
    )

    return DiagnosticsReport(
        scorer_name=(
            transition_scorer.name
        ),
        predictor_name=(
            predictor.name
        ),
        dialogue_count=(
            len(dialogue_reports)
        ),
        decision_count=(
            len(all_decisions)
        ),
        local_top1_accuracy=(
            local_top1
        ),
        local_mrr=local_mrr,
        mean_true_margin=(
            mean_margin
        ),
        mean_pairwise_score=(
            100.0 * mean_pairwise
        ),
        gold_path_beats_prediction_rate=(
            search_gap_rate
        ),
        dialogues=tuple(
            dialogue_reports
        ),
    )