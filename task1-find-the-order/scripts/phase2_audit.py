from __future__ import annotations

import json
import sys
from pathlib import Path


TASK_ROOT = Path(__file__).resolve().parents[1]

NOTEBOOK_DIR = TASK_ROOT / "notebooks"
SRC_DIR = TASK_ROOT / "src"
TESTS_DIR = TASK_ROOT / "tests"
NOTES_DIR = TASK_ROOT / "notes"

CONTEST_BASELINE = NOTEBOOK_DIR / "00_contest_baseline_original.ipynb"
EDUCATIONAL_BASELINE = NOTEBOOK_DIR / "01_educational_baseline.ipynb"
SOLUTION_DEV = NOTEBOOK_DIR / "02_solution_dev.ipynb"

REQUIRED_FILES = [
    TASK_ROOT / "statement.md",
    TASK_ROOT / "README.md",
    TASK_ROOT / "requirements-local.txt",
    SRC_DIR / "__init__.py",
    SRC_DIR / "order_utils.py",
    TESTS_DIR / "test_order_utils.py",
    NOTES_DIR / "baseline-notes.md",
    CONTEST_BASELINE,
    EDUCATIONAL_BASELINE,
    SOLUTION_DEV,
]


def print_header(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def validate_project_files() -> None:
    print_header("1. PROJECT FILES")

    missing_files: list[Path] = []

    for path in REQUIRED_FILES:
        relative = path.relative_to(TASK_ROOT)

        if path.is_file():
            print(f"[OK]      {relative}")
        else:
            print(f"[MISSING] {relative}")
            missing_files.append(path)

    if missing_files:
        print()
        print("Phase 2 is NOT complete.")
        print("Missing files:")

        for path in missing_files:
            print(f"  - {path.relative_to(TASK_ROOT)}")

        raise SystemExit(1)


def validate_scoring_utilities() -> None:
    print_header("2. SCORING UTILITIES")

    sys.path.insert(0, str(TASK_ROOT))

    from src.order_utils import (  # noqa: PLC0415
        order_to_rank,
        pairwise_score,
        rank_to_order,
        submission_score,
    )

    order = [1, 2, 0]
    rank = [2, 0, 1]

    assert order_to_rank(order) == rank
    assert rank_to_order(rank) == order

    assert pairwise_score(rank, rank) == 1.0

    reversed_rank = order_to_rank(list(reversed(order)))
    assert pairwise_score(reversed_rank, rank) == 0.0

    assert pairwise_score([0, 0, 1], rank) == 0.0

    score = submission_score(
        predictions={"a": rank},
        targets={
            "a": rank,
            "b": [0, 1, 2],
        },
    )

    assert score == 50.0

    print("[OK] order -> rank conversion")
    print("[OK] rank -> order conversion")
    print("[OK] perfect pairwise score")
    print("[OK] reversed pairwise score")
    print("[OK] invalid prediction handling")
    print("[OK] macro submission scoring")


def inspect_contest_baseline() -> None:
    print_header("3. OFFICIAL CONTEST BASELINE")

    notebook = json.loads(
        CONTEST_BASELINE.read_text(encoding="utf-8")
    )

    cells = notebook.get("cells", [])

    print(f"Notebook: {CONTEST_BASELINE.name}")
    print(f"Cells: {len(cells)}")

    source_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in cells
    )

    expected_markers = {
        "prefix.json": 'prefix.json',
        "answers.json": 'answers.json',
        "deterministic baseline": 'order = [first, second]',
        "Wav2Vec2 example": 'Wav2Vec2Model',
        "Qwen example": 'AutoModelForCausalLM',
        "Whisper example": 'WhisperForConditionalGeneration',
    }

    for label, marker in expected_markers.items():
        if marker not in source_text:
            raise AssertionError(
                f"Expected marker not found: {label!r} ({marker!r})"
            )

        print(f"[OK] {label}")

    print()
    print("CELL SUMMARY")
    print("-" * 80)

    for index, cell in enumerate(cells):
        cell_type = cell.get("cell_type", "unknown")
        source = "".join(cell.get("source", [])).strip()

        first_line = (
            source.splitlines()[0][:100]
            if source
            else "<empty>"
        )

        print(
            f"{index:02d} | "
            f"{cell_type:<8} | "
            f"{first_line}"
        )


def print_summary() -> None:
    print_header("PHASE 2 RESULT")

    print("All Phase 2 checks passed.")
    print()
    print("We now understand:")
    print("- input format")
    print("- prefix semantics")
    print("- rank convention")
    print("- official pairwise metric")
    print("- invalid prediction behavior")
    print("- contest baseline behavior")
    print("- model example cells")
    print("- local vs GPU responsibilities")
    print()
    print("Phase 2 is ready to be committed.")


def main() -> None:
    print(f"Task root: {TASK_ROOT}")

    validate_project_files()
    validate_scoring_utilities()
    inspect_contest_baseline()
    print_summary()


if __name__ == "__main__":
    main()