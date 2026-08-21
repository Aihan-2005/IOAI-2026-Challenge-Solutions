from src.memoized_scorer import (
    MemoizedTransitionScorer,
)


class CountingScorer:
    name = "counting"

    def __init__(
        self,
    ) -> None:
        self.processed_pairs = 0

    def score_many(
        self,
        pairs,
    ):
        self.processed_pairs += len(
            pairs
        )

        return [
            float(
                len(previous)
                + len(following)
            )
            for (
                previous,
                following,
            )
            in pairs
        ]

    def score(
        self,
        previous_text,
        next_text,
    ):
        return self.score_many(
            [
                (
                    previous_text,
                    next_text,
                )
            ]
        )[0]


def test_memoization_avoids_duplicate_work():
    base = CountingScorer()

    scorer = (
        MemoizedTransitionScorer(
            base
        )
    )

    pairs = [
        ("a", "b"),
        ("a", "c"),
        ("a", "b"),
    ]

    first = scorer.score_many(
        pairs
    )

    second = scorer.score_many(
        pairs
    )

    assert first == second

    assert (
        base.processed_pairs
        == 2
    )


def test_single_score_uses_same_cache():
    base = CountingScorer()

    scorer = (
        MemoizedTransitionScorer(
            base
        )
    )

    first = scorer.score(
        "hello",
        "world",
    )

    second = scorer.score(
        "hello",
        "world",
    )

    assert first == second

    assert (
        base.processed_pairs
        == 1
    )