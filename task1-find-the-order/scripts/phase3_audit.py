from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


TASK_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(TASK_ROOT),
)


from src.dataset_io import load_dialogues
from src.order_utils import is_valid_permutation
from src.pipeline import run_prediction_pipeline
from src.predictors import PrefixIndexBaseline
from src.submission import read_submission


def header(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def create_synthetic_split(
    root: Path,
) -> Path:
    split = root / "test_public"
    split.mkdir()

    prefixes = {
        "0": [1, 2],
        "7": [4, 1],
        "15": [0, 3],
    }

    (split / "prefix.json").write_text(
        json.dumps(prefixes),
        encoding="utf-8",
    )

    dialogue_sizes = {
        "0": 4,
        "7": 6,
        "15": 5,
    }

    for dialogue_id, n_chunks in (
        dialogue_sizes.items()
    ):
        dialogue_dir = split / dialogue_id
        dialogue_dir.mkdir()

        for index in range(n_chunks):
            (
                dialogue_dir
                / f"chunk_{index}.wav"
            ).write_bytes(b"")

    return split


def audit_dataset_loader(
    split: Path,
) -> None:
    header("1. DATASET LOADER")

    dialogues = load_dialogues(split)

    assert len(dialogues) == 3

    assert [
        dialogue.dialogue_id
        for dialogue in dialogues
    ] == [
        "0",
        "7",
        "15",
    ]

    for dialogue in dialogues:
        print(
            f"[OK] dialogue={dialogue.dialogue_id:<3} "
            f"chunks={dialogue.n_chunks:<2} "
            f"prefix={dialogue.prefix}"
        )


def audit_pipeline(
    split: Path,
    root: Path,
) -> None:
    header("2. END-TO-END PIPELINE")

    output = root / "answers.json"

    result = run_prediction_pipeline(
        split_dir=split,
        output_path=output,
        predictor=PrefixIndexBaseline(),
    )

    print(
        f"[OK] predictor: "
        f"{result.predictor_name}"
    )

    print(
        f"[OK] dialogues: "
        f"{result.dialogue_count}"
    )

    print(
        f"[OK] output: "
        f"{result.output_path.name}"
    )

    assert output.is_file()


def audit_submission(
    split: Path,
    root: Path,
) -> None:
    header("3. SUBMISSION VALIDATION")

    dialogues = load_dialogues(split)

    submission = read_submission(
        root / "answers.json"
    )

    assert set(submission) == {
        dialogue.dialogue_id
        for dialogue in dialogues
    }

    for dialogue in dialogues:
        prediction = submission[
            dialogue.dialogue_id
        ]

        assert is_valid_permutation(
            prediction,
            dialogue.n_chunks,
        )

        first, second = dialogue.prefix

        assert prediction[first] == 0
        assert prediction[second] == 1

        print(
            f"[OK] dialogue="
            f"{dialogue.dialogue_id:<3} "
            f"rank={prediction}"
        )


def main() -> None:
    header("PHASE 3 LOCAL PIPELINE AUDIT")

    print(f"Task root: {TASK_ROOT}")

    with tempfile.TemporaryDirectory() as tmp:
        temporary_root = Path(tmp)

        split = create_synthetic_split(
            temporary_root
        )

        audit_dataset_loader(split)

        audit_pipeline(
            split,
            temporary_root,
        )

        audit_submission(
            split,
            temporary_root,
        )

    header("PHASE 3 RESULT")

    print(
        "All Phase 3 local pipeline checks passed."
    )

    print()
    print("Validated:")
    print("- IOAI split discovery")
    print("- prefix.json parsing")
    print("- chunk discovery")
    print("- Dialogue domain objects")
    print("- Predictor interface")
    print("- official baseline predictor")
    print("- rank conversion")
    print("- answers.json generation")
    print("- hidden-test-safe inference")
    print()
    print(
        "The project is ready for "
        "real model integration."
    )


if __name__ == "__main__":
    main()