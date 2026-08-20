from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.domain import Dialogue


@dataclass(frozen=True, slots=True)
class ChunkTranscript:
    chunk_index: int
    text: str


@dataclass(frozen=True, slots=True)
class DialogueTranscript:
    dialogue_id: str
    chunks: tuple[ChunkTranscript, ...]

    def text_for(self, chunk_index: int) -> str:
        for chunk in self.chunks:
            if chunk.chunk_index == chunk_index:
                return chunk.text

        raise KeyError(
            f"No transcript for chunk {chunk_index}"
        )


class Transcriber(Protocol):
    """
    Interface for any audio-to-text backend.

    Future implementations:
    - Whisper
    - cached transcripts
    - mock transcriber for tests
    """

    name: str

    def transcribe(
        self,
        dialogue: Dialogue,
    ) -> DialogueTranscript:
        ...