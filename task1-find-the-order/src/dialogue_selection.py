from __future__ import annotations

import hashlib
from collections.abc import Sequence

from src.domain import Dialogue


def deterministic_dialogue_sample(
    dialogues: Sequence[Dialogue],
    *,
    count: int,
    seed: str,
) -> tuple[Dialogue, ...]:
    """
    Select a reproducible pseudo-random subset of dialogues.

    The selection is independent from Python's process-level
    random hash seed and therefore stable across machines.
    """
    if count < 1:
        raise ValueError(
            "count must be at least 1"
        )

    if count > len(dialogues):
        raise ValueError(
            f"Requested {count} dialogues, "
            f"but only {len(dialogues)} are available"
        )

    def selection_key(
        dialogue: Dialogue,
    ) -> tuple[bytes, str]:
        payload = (
            f"{seed}:{dialogue.dialogue_id}"
            .encode("utf-8")
        )

        digest = hashlib.sha256(
            payload
        ).digest()

        return (
            digest,
            dialogue.dialogue_id,
        )

    ordered = sorted(
        dialogues,
        key=selection_key,
    )

    return tuple(
        ordered[:count]
    )