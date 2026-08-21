from __future__ import annotations

from collections.abc import Sequence

from src.transition import (
    TransitionPair,
    TransitionScorer,
)


class MemoizedTransitionScorer:
    """
    In-memory cache around any transition scorer.

    This is particularly useful for expensive LM scorers because
    benchmark and diagnostics often request the same transitions.
    """

    def __init__(
        self,
        scorer: TransitionScorer,
    ) -> None:
        self._scorer = scorer

        self._cache: dict[
            TransitionPair,
            float,
        ] = {}

    @property
    def name(self) -> str:
        return self._scorer.name

    @property
    def cache_size(self) -> int:
        return len(
            self._cache
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
        pairs: Sequence[TransitionPair],
    ) -> list[float]:
        missing: list[
            TransitionPair
        ] = []

        seen_missing: set[
            TransitionPair
        ] = set()

        for pair in pairs:
            if pair in self._cache:
                continue

            if pair in seen_missing:
                continue

            seen_missing.add(
                pair
            )

            missing.append(
                pair
            )

        if missing:
            batch_method = getattr(
                self._scorer,
                "score_many",
                None,
            )

            if callable(batch_method):
                missing_scores = (
                    batch_method(
                        missing
                    )
                )

            else:
                missing_scores = [
                    self._scorer.score(
                        previous,
                        following,
                    )
                    for (
                        previous,
                        following,
                    )
                    in missing
                ]

            if (
                len(missing_scores)
                != len(missing)
            ):
                raise RuntimeError(
                    "Wrapped scorer returned "
                    "incorrect score count"
                )

            for pair, value in zip(
                missing,
                missing_scores,
                strict=True,
            ):
                self._cache[pair] = (
                    float(value)
                )

        return [
            self._cache[pair]
            for pair in pairs
        ]