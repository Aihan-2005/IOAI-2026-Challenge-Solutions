from src.heuristic_scorer import (
    HeuristicDialogueTransitionScorer,
)


def test_question_answer_transition():
    scorer = (
        HeuristicDialogueTransitionScorer()
    )

    question = (
        "Are you coming to the meeting?"
    )

    answer = (
        "No, I have a dentist appointment."
    )

    unrelated = (
        "The train station was renovated last year."
    )

    assert (
        scorer.score(
            question,
            answer,
        )
        >
        scorer.score(
            question,
            unrelated,
        )
    )


def test_why_because_transition():
    scorer = (
        HeuristicDialogueTransitionScorer()
    )

    previous = (
        "Why did you miss the meeting?"
    )

    strong_next = (
        "Because my train was delayed."
    )

    weak_next = (
        "The weather looks nice today."
    )

    assert (
        scorer.score(
            previous,
            strong_next,
        )
        >
        scorer.score(
            previous,
            weak_next,
        )
    )


def test_thanks_acknowledgement_transition():
    scorer = (
        HeuristicDialogueTransitionScorer()
    )

    previous = (
        "Thanks for sending me the notes."
    )

    strong_next = (
        "No problem, happy to help."
    )

    weak_next = (
        "I bought a new phone yesterday."
    )

    assert (
        scorer.score(
            previous,
            strong_next,
        )
        >
        scorer.score(
            previous,
            weak_next,
        )
    )