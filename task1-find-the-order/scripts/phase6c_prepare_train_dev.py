from __future__ import annotations

import argparse
import sys
from pathlib import Path


TASK_ROOT = (
    Path(__file__).resolve().parents[1]
)

sys.path.insert(
    0,
    str(TASK_ROOT),
)


from src.asr_whisper import (
    WhisperConfig,
    WhisperTranscriber,
)
from src.dataset_io import (
    load_dialogues,
)
from src.dialogue_selection import (
    deterministic_dialogue_sample,
)
from src.transcript_io import (
    TranscriptStore,
)
from src.transcription_runner import (
    populate_transcript_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--transcript-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--count",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--seed",
        default="phase6c-dev-v1",
    )

    parser.add_argument(
        "--model-id",
        default=(
            "openai/whisper-small"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dialogues = load_dialogues(
        args.split_dir
    )

    selected = (
        deterministic_dialogue_sample(
            dialogues,
            count=args.count,
            seed=args.seed,
        )
    )

    print(
        "Selected dialogue IDs:"
    )

    print(
        [
            dialogue.dialogue_id
            for dialogue in selected
        ]
    )

    store = TranscriptStore(
        args.transcript_dir
    )

    missing = [
        dialogue
        for dialogue in selected
        if not store.path_for(
            dialogue.dialogue_id
        ).is_file()
    ]

    print(
        "Already cached:",
        len(selected) - len(missing),
    )

    print(
        "Need transcription:",
        len(missing),
    )

    if not missing:
        print(
            "All selected transcripts "
            "are already cached."
        )

        return

    transcriber = WhisperTranscriber(
        WhisperConfig(
            model_id=args.model_id
        )
    )

    print(
        "ASR device:",
        transcriber.device,
    )

    stats = populate_transcript_cache(
        dialogues=selected,
        transcriber=transcriber,
        store=store,
    )

    print()
    print(stats)


if __name__ == "__main__":
    main()