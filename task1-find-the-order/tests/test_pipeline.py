import json

from src.pipeline import run_prediction_pipeline
from src.predictors import PrefixIndexBaseline
from src.submission import read_submission


def _make_dialogue(
    split,
    dialogue_id: str,
    n_chunks: int,
):
    directory = split / dialogue_id

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for index in range(n_chunks):
        (
            directory / f"chunk_{index}.wav"
        ).write_bytes(b"")


def test_pipeline_without_ground_truth(
    tmp_path,
):
    split = tmp_path / "test_public"
    split.mkdir()

    (split / "prefix.json").write_text(
        json.dumps(
            {
                "0": [1, 2],
                "1": [3, 0],
            }
        )
    )

    _make_dialogue(split, "0", 4)
    _make_dialogue(split, "1", 5)

    

    output = tmp_path / "answers.json"

    result = run_prediction_pipeline(
        split_dir=split,
        output_path=output,
        predictor=PrefixIndexBaseline(),
    )

    assert result.dialogue_count == 2

    assert result.predictor_name == (
        "prefix-index-baseline"
    )

    assert output.is_file()

    predictions = read_submission(output)

    assert set(predictions) == {
        "0",
        "1",
    }


def test_pipeline_output_is_valid_rank_format(
    tmp_path,
):
    split = tmp_path / "test_public"
    split.mkdir()

    (split / "prefix.json").write_text(
        json.dumps(
            {
                "7": [2, 0],
            }
        )
    )

    _make_dialogue(
        split,
        "7",
        4,
    )

    output = tmp_path / "answers.json"

    result = run_prediction_pipeline(
        split_dir=split,
        output_path=output,
        predictor=PrefixIndexBaseline(),
    )

    prediction = result.predictions["7"]

    assert sorted(prediction) == [
        0,
        1,
        2,
        3,
    ]

    assert prediction[2] == 0
    assert prediction[0] == 1