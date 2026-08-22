from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath


DEFAULT_DATASET_ID = (
    "IOAI-official/ioai-2026-find-the-order"
)


class DatasetDownloadError(RuntimeError):
    """Raised when an official dataset subset cannot be resolved."""


@dataclass(frozen=True, slots=True)
class DownloadedSubset:
    dataset_root: Path
    split_dir: Path
    answers_path: Path | None


def _find_answers_path(
    dataset_root: Path,
    subset: str,
) -> Path | None:
    subset_path = PurePosixPath(
        subset
    )

    parent = (
        dataset_root
        / subset_path.parent
    )

    name = subset_path.name

    exact = (
        parent
        / f"{name}_answers.json"
    )

    if exact.is_file():
        return exact

    candidates = sorted(
        parent.glob(
            f"{name}*answers*.json"
        )
    )

    if not candidates:
        return None

    if len(candidates) > 1:
        raise DatasetDownloadError(
            "Multiple possible answer files found: "
            + ", ".join(
                str(path)
                for path in candidates
            )
        )

    return candidates[0]


def resolve_local_subset(
    *,
    dataset_root: str | Path,
    subset: str,
    require_answers: bool = False,
) -> DownloadedSubset:
    """
    Resolve an already-downloaded subset without making
    any network request.

    This is intentionally separate from downloading so that
    Colab experiments can recover cleanly after transient
    network failures.
    """
    root = Path(
        dataset_root
    ).expanduser().resolve()

    split_dir = (
        root
        / PurePosixPath(subset)
    )

    if not split_dir.is_dir():
        raise DatasetDownloadError(
            f"Dataset subset does not exist locally: "
            f"{split_dir}"
        )

    answers_path = (
        _find_answers_path(
            root,
            subset,
        )
    )

    if (
        require_answers
        and answers_path is None
    ):
        raise DatasetDownloadError(
            f"Answer file was not found for "
            f"subset {subset!r}"
        )

    return DownloadedSubset(
        dataset_root=root,
        split_dir=split_dir,
        answers_path=answers_path,
    )


def download_hf_subset(
    *,
    subset: str = "public/pretest",
    destination: str | Path,
    dataset_id: str = DEFAULT_DATASET_ID,
    max_workers: int = 4,
    require_answers: bool = False,
) -> DownloadedSubset:
    """
    Download exactly one subset of the official IOAI dataset.

    The Hugging Face dependency is imported lazily so that this
    module remains usable in the lightweight local environment.

    Existing partial downloads can be reused by subsequent calls.
    """
    if max_workers < 1:
        raise ValueError(
            "max_workers must be at least 1"
        )

    from huggingface_hub import (
        snapshot_download,
    )

    destination_path = Path(
        destination
    ).expanduser().resolve()

    destination_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    subset_path = PurePosixPath(
        subset
    )

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
            max_workers=max_workers,
        )
    )

    return resolve_local_subset(
        dataset_root=snapshot_path,
        subset=subset,
        require_answers=require_answers,
    )