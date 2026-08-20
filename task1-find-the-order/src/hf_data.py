from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath


DEFAULT_DATASET_ID = (
    "IOAI-official/ioai-2026-find-the-order"
)


@dataclass(frozen=True, slots=True)
class DownloadedSubset:
    dataset_root: Path
    split_dir: Path
    answers_path: Path | None


def _find_answers_path(
    dataset_root: Path,
    subset: str,
) -> Path | None:
    subset_path = PurePosixPath(subset)

    parent = dataset_root / subset_path.parent
    name = subset_path.name

    exact = parent / f"{name}_answers.json"

    if exact.is_file():
        return exact

    candidates = sorted(
        parent.glob(f"{name}*answers*.json")
    )

    if not candidates:
        return None

    if len(candidates) > 1:
        raise RuntimeError(
            "Multiple possible answer files found: "
            + ", ".join(
                str(path)
                for path in candidates
            )
        )

    return candidates[0]


def download_hf_subset(
    *,
    subset: str = "public/pretest",
    destination: str | Path,
    dataset_id: str = DEFAULT_DATASET_ID,
) -> DownloadedSubset:
    """
    Download only one subset of the official IOAI dataset.

    Heavy dependencies are imported lazily so that this module
    remains importable in the lightweight Mac development
    environment.
    """
    from huggingface_hub import snapshot_download

    destination_path = Path(
        destination
    ).expanduser().resolve()

    destination_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    subset_path = PurePosixPath(subset)

    answer_pattern = (
        f"{subset_path.parent}/"
        f"{subset_path.name}*answers*.json"
    )

    snapshot_path = Path(
        snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            local_dir=destination_path,
            allow_patterns=[
                f"{subset}/**",
                answer_pattern,
            ],
        )
    )

    split_dir = snapshot_path / subset

    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"Downloaded subset not found: {split_dir}"
        )

    answers_path = _find_answers_path(
        snapshot_path,
        subset,
    )

    return DownloadedSubset(
        dataset_root=snapshot_path,
        split_dir=split_dir,
        answers_path=answers_path,
    )