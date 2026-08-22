from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path

from src.transition import (
    TransitionPair,
    TransitionScorer,
)


class SQLiteCachedTransitionScorer:
    """
    Persistent transition-score cache.

    Designed for expensive scorers such as language models.

    Cache entries survive:
    - Python process restarts
    - notebook cell reruns
    - repeated diagnostics
    - repeated benchmark runs
    """

    def __init__(
        self,
        *,
        scorer: TransitionScorer,
        database_path: str | Path,
        namespace: str | None = None,
    ) -> None:
        self._scorer = scorer

        self._database_path = Path(
            database_path
        )

        self._database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._namespace = (
            namespace
            or scorer.name
        )

        self._initialize_database()

    @property
    def name(self) -> str:
        return self._scorer.name

    @property
    def database_path(self) -> Path:
        return self._database_path

    def _connect(
        self,
    ) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self._database_path
        )

        connection.execute(
            "PRAGMA journal_mode=WAL"
        )

        connection.execute(
            "PRAGMA synchronous=NORMAL"
        )

        return connection

    def _initialize_database(
        self,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS transition_scores (
                    namespace TEXT NOT NULL,
                    previous_text TEXT NOT NULL,
                    next_text TEXT NOT NULL,
                    score REAL NOT NULL,

                    PRIMARY KEY (
                        namespace,
                        previous_text,
                        next_text
                    )
                )
                """
            )

    def score(
        self,
        previous_text: str,
        next_text: str,
    ) -> float:
        return self.score_many(
            [
                (
                    previous_text,
                    next_text,
                )
            ]
        )[0]

    def score_many(
        self,
        pairs: Sequence[
            TransitionPair
        ],
    ) -> list[float]:
        if not pairs:
            return []

        unique_pairs = list(
            dict.fromkeys(
                pairs
            )
        )

        cached: dict[
            TransitionPair,
            float,
        ] = {}

        missing: list[
            TransitionPair
        ] = []

        with self._connect() as connection:
            cursor = connection.cursor()

            for (
                previous_text,
                next_text,
            ) in unique_pairs:
                row = cursor.execute(
                    """
                    SELECT score
                    FROM transition_scores
                    WHERE namespace = ?
                      AND previous_text = ?
                      AND next_text = ?
                    """,
                    (
                        self._namespace,
                        previous_text,
                        next_text,
                    ),
                ).fetchone()

                pair = (
                    previous_text,
                    next_text,
                )

                if row is None:
                    missing.append(
                        pair
                    )
                else:
                    cached[pair] = float(
                        row[0]
                    )

        if missing:
            batch_method = getattr(
                self._scorer,
                "score_many",
                None,
            )

            if callable(
                batch_method
            ):
                missing_scores = (
                    batch_method(
                        missing
                    )
                )

            else:
                missing_scores = [
                    self._scorer.score(
                        previous_text,
                        next_text,
                    )
                    for (
                        previous_text,
                        next_text,
                    )
                    in missing
                ]

            if (
                len(missing_scores)
                != len(missing)
            ):
                raise RuntimeError(
                    "Wrapped scorer returned "
                    "an invalid number of scores"
                )

            rows_to_write = []

            for pair, score in zip(
                missing,
                missing_scores,
                strict=True,
            ):
                value = float(
                    score
                )

                cached[pair] = value

                rows_to_write.append(
                    (
                        self._namespace,
                        pair[0],
                        pair[1],
                        value,
                    )
                )

            with self._connect() as connection:
                connection.executemany(
                    """
                    INSERT OR REPLACE INTO
                    transition_scores (
                        namespace,
                        previous_text,
                        next_text,
                        score
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    rows_to_write,
                )

        return [
            cached[pair]
            for pair in pairs
        ]

    @property
    def cache_size(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*)
                FROM transition_scores
                WHERE namespace = ?
                """,
                (
                    self._namespace,
                ),
            ).fetchone()

        if row is None:
            return 0

        return int(
            row[0]
        )

