from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

from src.domain import Dialogue
from src.transcript_io import TranscriptStore
from src.transcription import Transcriber


@dataclass(frozen=True, slots=True)
class TranscriptionStats:
    total_dialogues: int
    transcribed_dialogues: int
    cached_dialogues: int
    elapsed_seconds: float


def populate_transcript_cache(
    *,
    dialogues: Iterable[Dialogue],
    transcriber: Transcriber,
    store: TranscriptStore,
    verbose: bool = True,
) -> TranscriptionStats:
    """
    Transcribe missing dialogues and reuse existing cached artifacts.

    Safe to rerun after interruption.
    """
    start = perf_counter()

    dialogue_list = list(dialogues)

    transcribed = 0
    cached = 0

    for position, dialogue in enumerate(
        dialogue_list,
        start=1,
    ):
        output_path = store.path_for(
            dialogue.dialogue_id
        )

        if output_path.is_file():
            cached += 1

            if verbose:
                print(
                    f"[{position}/{len(dialogue_list)}] "
                    f"[CACHE] dialogue={dialogue.dialogue_id}"
                )

            continue

        if verbose:
            print(
                f"[{position}/{len(dialogue_list)}] "
                f"[ASR] dialogue={dialogue.dialogue_id} "
                f"chunks={dialogue.n_chunks}"
            )

        transcript = transcriber.transcribe(
            dialogue
        )

        store.save(
            transcript
        )

        transcribed += 1

    elapsed = perf_counter() - start

    return TranscriptionStats(
        total_dialogues=len(dialogue_list),
        transcribed_dialogues=transcribed,
        cached_dialogues=cached,
        elapsed_seconds=elapsed,
    )