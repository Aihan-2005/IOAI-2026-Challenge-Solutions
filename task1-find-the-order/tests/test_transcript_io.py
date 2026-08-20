import json

import pytest

from src.transcript_io import (
    TranscriptFormatError,
    TranscriptStore,
)
from src.transcription import (
    ChunkTranscript,
    DialogueTranscript,
)


def test_transcript_round_trip(
    tmp_path,
):
    store = TranscriptStore(
        tmp_path / "transcripts"
    )

    original = DialogueTranscript(
        dialogue_id="42",
        chunks=(
            ChunkTranscript(
                chunk_index=0,
                text="Hello.",
            ),
            ChunkTranscript(
                chunk_index=1,
                text="Hi, how are you?",
            ),
            ChunkTranscript(
                chunk_index=2,
                text="I'm doing well.",
            ),
        ),
    )

    output = store.save(
        original
    )

    assert output.is_file()

    loaded = store.load(
        "42"
    )

    assert loaded == original


def test_non_contiguous_transcript_rejected(
    tmp_path,
):
    root = (
        tmp_path
        / "transcripts"
    )

    root.mkdir()

    path = root / "7.json"

    path.write_text(
        json.dumps(
            {
                "dialogue_id": "7",
                "chunks": [
                    {
                        "chunk_index": 0,
                        "text": "First",
                    },
                    {
                        "chunk_index": 2,
                        "text": "Third",
                    },
                ],
            }
        )
    )

    store = TranscriptStore(
        root
    )

    with pytest.raises(
        TranscriptFormatError,
        match="contiguous",
    ):
        store.load(
            "7"
        )