from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from src.dataset_io import load_dialogues
from src.predictors import RankingPredictor
from src.submission import write_submission


@dataclass(frozen=True, slots=True)
class PipelineResult:
    predictor_name: str
    dialogue_count: int
    elapsed_seconds: float
    predictions: dict[str, list[int]]
    output_path: Path


def run_prediction_pipeline(
    *,
    split_dir: str | Path,
    output_path: str | Path,
    predictor: RankingPredictor,
) -> PipelineResult:
    """
    Execute a complete Task 1 inference pipeline.

    Dataset
        -> load dialogues
        -> predictor
        -> validation
        -> answers.json
    """
    start = perf_counter()

    dialogues = load_dialogues(split_dir)

    predictions: dict[str, list[int]] = {}

    for dialogue in dialogues:
        predictions[dialogue.dialogue_id] = (
            predictor.predict(dialogue)
        )

    written_path = write_submission(
        output_path=output_path,
        dialogues=dialogues,
        predictions=predictions,
    )

    elapsed = perf_counter() - start

    return PipelineResult(
        predictor_name=predictor.name,
        dialogue_count=len(dialogues),
        elapsed_seconds=elapsed,
        predictions=predictions,
        output_path=written_path,
    )