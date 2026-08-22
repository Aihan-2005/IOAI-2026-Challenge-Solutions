import json

import pytest

from src.hf_data import (
    DatasetDownloadError,
    resolve_local_subset,
)


def test_resolve_existing_subset(
    tmp_path,
):
    split = (
        tmp_path
        / "public"
        / "train"
    )

    split.mkdir(
        parents=True
    )

    (
        split
        / "prefix.json"
    ).write_text(
        json.dumps(
            {
                "0": [0, 1],
            }
        ),
        encoding="utf-8",
    )

    answers = (
        tmp_path
        / "public"
        / "train_answers.json"
    )

    answers.write_text(
        json.dumps(
            {
                "0": [0, 1],
            }
        ),
        encoding="utf-8",
    )

    result = resolve_local_subset(
        dataset_root=tmp_path,
        subset="public/train",
        require_answers=True,
    )

    assert (
        result.split_dir
        == split
    )

    assert (
        result.answers_path
        == answers
    )


def test_missing_subset_is_rejected(
    tmp_path,
):
    with pytest.raises(
        DatasetDownloadError,
        match="does not exist locally",
    ):
        resolve_local_subset(
            dataset_root=tmp_path,
            subset="public/train",
        )


def test_required_answers_are_checked(
    tmp_path,
):
    split = (
        tmp_path
        / "public"
        / "train"
    )

    split.mkdir(
        parents=True
    )

    with pytest.raises(
        DatasetDownloadError,
        match="Answer file was not found",
    ):
        resolve_local_subset(
            dataset_root=tmp_path,
            subset="public/train",
            require_answers=True,
        )

