from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.transcription import (
    ChunkTranscript,
    DialogueTranscript,
)


class TranscriptFormatError(
    ValueError
):
    """Raised when transcript data is malformed."""


def _validate_transcript(
    transcript: DialogueTranscript,
) -> None:
    indexes = [
        chunk.chunk_index
        for chunk in transcript.chunks
    ]

    expected = list(
        range(len(indexes))
    )

    if sorted(indexes) != expected:
        raise TranscriptFormatError(
            "transcript chunk indexes must be "
            f"a contiguous permutation of {expected}"
        )

    for chunk in transcript.chunks:
        if not isinstance(
            chunk.text,
            str,
        ):
            raise TranscriptFormatError(
                "chunk text must be a string"
            )


@dataclass(frozen=True, slots=True)
class TranscriptStore:
    """
    Persistent store for dialogue transcripts.

    Intended workflow:

        GPU:
            WAV -> Whisper -> JSON

        Mac:
            JSON -> ordering experiments
    """

    root: Path

    def __init__(
        self,
        root: str | Path,
    ) -> None:
        object.__setattr__(
            self,
            "root",
            Path(root),
        )

    def path_for(
        self,
        dialogue_id: str,
    ) -> Path:
        return (
            self.root
            / f"{dialogue_id}.json"
        )

    def save(
        self,
        transcript: DialogueTranscript,
    ) -> Path:
        _validate_transcript(
            transcript
        )

        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = self.path_for(
            transcript.dialogue_id
        )

        temporary_path = (
            output_path.with_suffix(
                ".json.tmp"
            )
        )

        payload = {
            "dialogue_id": (
                transcript.dialogue_id
            ),
            "chunks": [
                {
                    "chunk_index": (
                        chunk.chunk_index
                    ),
                    "text": chunk.text,
                }
                for chunk in transcript.chunks
            ],
        }

        temporary_path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        temporary_path.replace(
            output_path
        )

        return output_path

    def load(
        self,
        dialogue_id: str,
    ) -> DialogueTranscript:
        path = self.path_for(
            dialogue_id
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"Transcript not found: {path}"
            )

        try:
            payload = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )
        except json.JSONDecodeError as exc:
            raise TranscriptFormatError(
                f"Invalid transcript JSON: {path}"
            ) from exc

        if not isinstance(
            payload,
            dict,
        ):
            raise TranscriptFormatError(
                "transcript JSON root must be an object"
            )

        stored_dialogue_id = payload.get(
            "dialogue_id"
        )

        if stored_dialogue_id != dialogue_id:
            raise TranscriptFormatError(
                "dialogue_id does not match filename"
            )

        raw_chunks = payload.get(
            "chunks"
        )

        if not isinstance(
            raw_chunks,
            list,
        ):
            raise TranscriptFormatError(
                "chunks must be a JSON list"
            )

        chunks: list[
            ChunkTranscript
        ] = []

        for raw_chunk in raw_chunks:
            if not isinstance(
                raw_chunk,
                dict,
            ):
                raise TranscriptFormatError(
                    "each chunk must be an object"
                )

            chunk_index = raw_chunk.get(
                "chunk_index"
            )

            text = raw_chunk.get(
                "text"
            )

            if (
                not isinstance(
                    chunk_index,
                    int,
                )
                or isinstance(
                    chunk_index,
                    bool,
                )
            ):
                raise TranscriptFormatError(
                    "chunk_index must be an integer"
                )

            if not isinstance(
                text,
                str,
            ):
                raise TranscriptFormatError(
                    "text must be a string"
                )

            chunks.append(
                ChunkTranscript(
                    chunk_index=chunk_index,
                    text=text,
                )
            )

        transcript = DialogueTranscript(
            dialogue_id=dialogue_id,
            chunks=tuple(chunks),
        )

        _validate_transcript(
            transcript
        )

        return transcript