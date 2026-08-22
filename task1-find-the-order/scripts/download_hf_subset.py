from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


TASK_ROOT = (
    Path(__file__).resolve().parents[1]
)

sys.path.insert(
    0,
    str(TASK_ROOT),
)


from src.hf_data import (
    download_hf_subset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Robust downloader for one IOAI "
            "Hugging Face dataset subset."
        )
    )

    parser.add_argument(
        "--subset",
        required=True,
    )

    parser.add_argument(
        "--destination",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--attempts",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--initial-delay",
        type=float,
        default=20.0,
    )

    parser.add_argument(
        "--require-answers",
        action="store_true",
    )

    return parser.parse_args()


def is_rate_limit_error(
    exception: BaseException,
) -> bool:
    message = str(
        exception
    ).lower()

    return (
        "429" in message
        or "too many requests" in message
    )


def main() -> None:
    args = parse_args()

    if args.attempts < 1:
        raise ValueError(
            "attempts must be at least 1"
        )

    for attempt in range(
        1,
        args.attempts + 1,
    ):
        print()
        print(
            f"Download attempt "
            f"{attempt}/{args.attempts}"
        )

        try:
            result = (
                download_hf_subset(
                    subset=args.subset,
                    destination=(
                        args.destination
                    ),
                    max_workers=(
                        args.max_workers
                    ),
                    require_answers=(
                        args.require_answers
                    ),
                )
            )

        except Exception as exc:
            if (
                not is_rate_limit_error(
                    exc
                )
                or attempt
                == args.attempts
            ):
                raise

            delay = (
                args.initial_delay
                * (2 ** (attempt - 1))
            )

            print(
                "Hugging Face rate limit "
                "detected."
            )

            print(
                f"Retrying in "
                f"{delay:.0f} seconds..."
            )

            time.sleep(
                delay
            )

            continue

        print()
        print(
            "DOWNLOAD COMPLETE"
        )

        print(
            "Dataset root:",
            result.dataset_root,
        )

        print(
            "Split:",
            result.split_dir,
        )

        print(
            "Answers:",
            result.answers_path,
        )

        return


if __name__ == "__main__":
    main()