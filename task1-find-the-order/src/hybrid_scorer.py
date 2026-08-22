from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from math import sqrt

from src.transition import (
    TransitionPair,
    TransitionScorer,
)


def _score_many(
    scorer: TransitionScorer,
    pairs: Sequence[TransitionPair],
) -> list[float]:
    batch_method = getattr(
        scorer,
        "score_many",
        None,
    )

    if callable(batch_method):
        values = batch_method(
            pairs
        )

    else:
        values = [
            scorer.score(
                previous,
                following,
            )
            for (
                previous,
                following,
            )
            in pairs
        ]

    if len(values) != len(pairs):
        raise RuntimeError(
            "Scorer returned an invalid "
            "number of transition scores"
        )

    return [
        float(value)
        for value in values
    ]


def _zscore(
    values: Sequence[float],
) -> list[float]:
    if not values:
        return []

    mean = (
        sum(values)
        / len(values)
    )

    variance = (
        sum(
            (value - mean) ** 2
            for value in values
        )
        / len(values)
    )

    standard_deviation = sqrt(
        variance
    )

    if standard_deviation < 1e-8:
        return [
            0.0
            for _ in values
        ]

    return [
        (
            value - mean
        )
        / standard_deviation
        for value in values
    ]


class HybridTransitionScorer:
    """
    Fuse heuristic and semantic transition rankings.

    Scores are normalized independently for every previous
    dialogue turn before fusion.

    This avoids directly mixing two incompatible numeric scales.
    """

    def __init__(
        self,
        *,
        heuristic_scorer: TransitionScorer,
        semantic_scorer: TransitionScorer,
        semantic_weight: float,
    ) -> None:
        if not (
            0.0
            <= semantic_weight
            <= 1.0
        ):
            raise ValueError(
                "semantic_weight must be in [0, 1]"
            )

        self.heuristic_scorer = (
            heuristic_scorer
        )

        self.semantic_scorer = (
            semantic_scorer
        )

        self.semantic_weight = (
            semantic_weight
        )

    @property
    def name(self) -> str:
        return (
            "hybrid-zscore-"
            f"semantic-{self.semantic_weight:.2f}"
        )

    def score_many(
        self,
        pairs: Sequence[TransitionPair],
    ) -> list[float]:
        if not pairs:
            return []

        heuristic_scores = (
            _score_many(
                self.heuristic_scorer,
                pairs,
            )
        )

        semantic_scores = (
            _score_many(
                self.semantic_scorer,
                pairs,
            )
        )

        groups: dict[
            str,
            list[int],
        ] = defaultdict(list)

        for index, (
            previous_text,
            _,
        ) in enumerate(
            pairs
        ):
            groups[
                previous_text
            ].append(
                index
            )

        fused = [
            0.0
            for _ in pairs
        ]

        semantic_weight = (
            self.semantic_weight
        )

        heuristic_weight = (
            1.0
            - semantic_weight
        )

        for indexes in groups.values():
            local_heuristic = [
                heuristic_scores[index]
                for index in indexes
            ]

            local_semantic = [
                semantic_scores[index]
                for index in indexes
            ]

            normalized_heuristic = (
                _zscore(
                    local_heuristic
                )
            )

            normalized_semantic = (
                _zscore(
                    local_semantic
                )
            )

            for local_position, (
                global_index
            ) in enumerate(
                indexes
            ):
                fused[
                    global_index
                ] = (
                    heuristic_weight
                    * normalized_heuristic[
                        local_position
                    ]
                    +
                    semantic_weight
                    * normalized_semantic[
                        local_position
                    ]
                )

        return fused

    def score(
        self,
        previous_text: str,
        next_text: str,
    ) -> float:
        """
        Scalar fallback.

        Production matrix construction uses score_many(),
        where per-source normalization is available.
        """
        heuristic = (
            self.heuristic_scorer
            .score(
                previous_text,
                next_text,
            )
        )

        semantic = (
            self.semantic_scorer
            .score(
                previous_text,
                next_text,
            )
        )

        return (
            (
                1.0
                - self.semantic_weight
            )
            * heuristic
            +
            self.semantic_weight
            * semantic
        )