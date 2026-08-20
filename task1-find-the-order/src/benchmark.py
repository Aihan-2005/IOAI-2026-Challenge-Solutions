from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping

from src.domain import Dialogue
from src.order_utils import pairwise_score
from src.predictors import RankingPredictor


@dataclass(frozen=True, slots=True)
class DialogueBenchmarkResult:
    dialogue_id: str
    n_chunks: int
    prefix: tuple[int, int]
    score: float
    prediction: list[int]
    target: list[int]


@dataclass(frozen=True, slots=True)
class PredictorBenchmark:
    predictor_name: str
    mean_score: float
    dialogue_count: int
    results: tuple[DialogueBenchmarkResult, ...]


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    created_at_utc: str
    dataset_name: str
    predictors: tuple[PredictorBenchmark, ...]


def evaluate_predictor(
    *,
    dialogues: tuple[Dialogue, ...],
    targets: Mapping[str, list[int]],
    predictor: RankingPredictor,
) -> PredictorBenchmark:
    results: list[
        DialogueBenchmarkResult
    ] = []

    for dialogue in dialogues:
        if dialogue.dialogue_id not in targets:
            raise KeyError(
                f"Missing target for dialogue "
                f"{dialogue.dialogue_id}"
            )

        target = targets[
            dialogue.dialogue_id
        ]

        prediction = predictor.predict(
            dialogue
        )

        score = pairwise_score(
            prediction,
            target,
        )

        results.append(
            DialogueBenchmarkResult(
                dialogue_id=dialogue.dialogue_id,
                n_chunks=dialogue.n_chunks,
                prefix=dialogue.prefix,
                score=score,
                prediction=prediction,
                target=target,
            )
        )

    mean_score = (
        100.0
        * sum(
            result.score
            for result in results
        )
        / len(results)
    )

    return PredictorBenchmark(
        predictor_name=predictor.name,
        mean_score=mean_score,
        dialogue_count=len(results),
        results=tuple(results),
    )


def build_report(
    *,
    dataset_name: str,
    benchmarks: tuple[
        PredictorBenchmark,
        ...,
    ],
) -> BenchmarkReport:
    return BenchmarkReport(
        created_at_utc=(
            datetime.now(UTC)
            .isoformat()
        ),
        dataset_name=dataset_name,
        predictors=benchmarks,
    )


def report_to_dict(
    report: BenchmarkReport,
) -> dict:
    return asdict(report)