from src.hybrid_scorer import (
    HybridTransitionScorer,
)


class HeuristicTestScorer:
    name = "heuristic-test"

    def score(
        self,
        previous_text: str,
        next_text: str,
    ) -> float:
        scores = {
            "a": 10.0,
            "b": 5.0,
            "c": 0.0,
        }

        return scores[
            next_text
        ]


class SemanticTestScorer:
    name = "semantic-test"

    def score(
        self,
        previous_text: str,
        next_text: str,
    ) -> float:
        scores = {
            "a": 0.0,
            "b": 5.0,
            "c": 10.0,
        }

        return scores[
            next_text
        ]


def test_low_semantic_weight_prefers_heuristic():
    scorer = HybridTransitionScorer(
        heuristic_scorer=(
            HeuristicTestScorer()
        ),
        semantic_scorer=(
            SemanticTestScorer()
        ),
        semantic_weight=0.25,
    )

    pairs = [
        ("previous", "a"),
        ("previous", "b"),
        ("previous", "c"),
    ]

    scores = scorer.score_many(
        pairs
    )

    best_index = max(
        range(len(scores)),
        key=scores.__getitem__,
    )

    assert (
        pairs[best_index][1]
        == "a"
    )


def test_high_semantic_weight_prefers_semantic():
    scorer = HybridTransitionScorer(
        heuristic_scorer=(
            HeuristicTestScorer()
        ),
        semantic_scorer=(
            SemanticTestScorer()
        ),
        semantic_weight=0.75,
    )

    pairs = [
        ("previous", "a"),
        ("previous", "b"),
        ("previous", "c"),
    ]

    scores = scorer.score_many(
        pairs
    )

    best_index = max(
        range(len(scores)),
        key=scores.__getitem__,
    )

    assert (
        pairs[best_index][1]
        == "c"
    )


def test_hybrid_scores_are_deterministic():
    scorer = HybridTransitionScorer(
        heuristic_scorer=(
            HeuristicTestScorer()
        ),
        semantic_scorer=(
            SemanticTestScorer()
        ),
        semantic_weight=0.5,
    )

    pairs = [
        ("previous", "a"),
        ("previous", "b"),
        ("previous", "c"),
    ]

    first = scorer.score_many(
        pairs
    )

    second = scorer.score_many(
        pairs
    )

    assert first == second