from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Dialogue:
    """
    Immutable representation of one IOAI dialogue.

    `chunk_paths[index]` always points to `chunk_{index}.wav`.
    """

    dialogue_id: str
    directory: Path
    chunk_paths: tuple[Path, ...]
    prefix: tuple[int, int]

    @property
    def n_chunks(self) -> int:
        return len(self.chunk_paths)

    @property
    def first_chunk(self) -> int:
        return self.prefix[0]

    @property
    def second_chunk(self) -> int:
        return self.prefix[1]

    def chunk_path(self, chunk_index: int) -> Path:
        if chunk_index < 0 or chunk_index >= self.n_chunks:
            raise IndexError(
                f"chunk index {chunk_index} is outside "
                f"0..{self.n_chunks - 1}"
            )

        return self.chunk_paths[chunk_index]
    