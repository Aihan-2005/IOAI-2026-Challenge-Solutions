from __future__ import annotations

from src.domain import Dialogue
from src.order_utils import (
    rank_to_order,
)
from src.transcript_io import (
    TranscriptStore,
)


def format_dialogue_failure(
    *,
    dialogue: Dialogue,
    target_rank: list[int],
    predicted_rank: list[int],
    transcript_store: TranscriptStore,
) -> str:
    transcript = (
        transcript_store.load(
            dialogue.dialogue_id
        )
    )

    truth_order = rank_to_order(
        target_rank
    )

    predicted_order = rank_to_order(
        predicted_rank
    )

    lines: list[str] = []

    lines.append(
        "=" * 80
    )

    lines.append(
        f"Dialogue "
        f"{dialogue.dialogue_id}"
    )

    lines.append(
        f"Chunks: "
        f"{dialogue.n_chunks}"
    )

    lines.append(
        f"Prefix: "
        f"{dialogue.prefix}"
    )

    lines.append("")

    lines.append(
        "TRUE ORDER:"
    )

    for position, chunk_index in (
        enumerate(
            truth_order
        )
    ):
        text = transcript.text_for(
            chunk_index
        )

        lines.append(
            f"{position:02d}. "
            f"chunk_{chunk_index:<2} "
            f"| {text}"
        )

    lines.append("")
    lines.append(
        "PREDICTED ORDER:"
    )

    for position, chunk_index in (
        enumerate(
            predicted_order
        )
    ):
        text = transcript.text_for(
            chunk_index
        )

        lines.append(
            f"{position:02d}. "
            f"chunk_{chunk_index:<2} "
            f"| {text}"
        )

    lines.append(
        "=" * 80
    )

    return "\n".join(
        lines
    )