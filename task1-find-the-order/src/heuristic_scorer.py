from __future__ import annotations

import re


_WORD_PATTERN = re.compile(
    r"[a-zA-Z']+"
)


_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "him",
        "his",
        "i",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "she",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "with",
        "you",
        "your",
    }
)


def _normalize(
    text: str,
) -> str:
    return " ".join(
        text.lower().strip().split()
    )


def _tokens(
    text: str,
) -> set[str]:
    return {
        token
        for token in _WORD_PATTERN.findall(
            text.lower()
        )
        if token not in _STOPWORDS
    }


def _starts_with(
    text: str,
    phrases: tuple[str, ...],
) -> bool:
    normalized = _normalize(text)

    return any(
        normalized == phrase
        or normalized.startswith(
            phrase + " "
        )
        or normalized.startswith(
            phrase + ","
        )
        for phrase in phrases
    )


def _contains(
    text: str,
    phrases: tuple[str, ...],
) -> bool:
    normalized = _normalize(text)

    return any(
        phrase in normalized
        for phrase in phrases
    )


class HeuristicDialogueTransitionScorer:
    """
    Lightweight discourse-aware transition scorer.

    This is intentionally dependency-free.

    It is not intended to be the final competition model.
    Its purpose is to make the ordering engine testable before
    GPU model integration.
    """

    name = "heuristic-dialogue-v1"

    def score(
        self,
        previous_text: str,
        next_text: str,
    ) -> float:
        previous = _normalize(
            previous_text
        )
        following = _normalize(
            next_text
        )

        if not previous or not following:
            return -5.0

        score = 0.0



        previous_tokens = _tokens(
            previous
        )

        following_tokens = _tokens(
            following
        )

        union = (
            previous_tokens
            | following_tokens
        )

        if union:
            overlap = (
                previous_tokens
                & following_tokens
            )

            jaccard = (
                len(overlap)
                / len(union)
            )

            score += 1.5 * jaccard



        if previous.endswith("?"):
            score += 0.75

            if _starts_with(
                following,
                (
                    "yes",
                    "yeah",
                    "yep",
                    "no",
                    "nope",
                    "sure",
                    "maybe",
                    "probably",
                    "actually",
                    "i",
                    "we",
                    "he",
                    "she",
                    "they",
                    "it",
                ),
            ):
                score += 1.5


                

        if _starts_with(
            previous,
            (
                "why",
                "how come",
            ),
        ) and _starts_with(
            following,
            (
                "because",
                "since",
            ),
        ):
            score += 3.0



        if _contains(
            previous,
            (
                "thank you",
                "thanks",
            ),
        ) and _starts_with(
            following,
            (
                "you're welcome",
                "you are welcome",
                "no problem",
                "no worries",
                "of course",
                "anytime",
            ),
        ):
            score += 4.0

    

        if _starts_with(
            previous,
            (
                "hello",
                "hi",
                "hey",
                "good morning",
                "good afternoon",
                "good evening",
            ),
        ) and _starts_with(
            following,
            (
                "hello",
                "hi",
                "hey",
                "good morning",
                "good afternoon",
                "good evening",
            ),
        ):
            score += 2.0


       
        if _starts_with(
            following,
            (
                "okay",
                "ok",
                "right",
                "exactly",
                "great",
                "sure",
                "well",
                "so",
            ),
        ):
            score += 0.5




        previous_is_farewell = _contains(
            previous,
            (
                "goodbye",
                "bye",
                "see you",
                "talk to you later",
            ),
        )

        following_is_farewell = _contains(
            following,
            (
                "goodbye",
                "bye",
                "see you",
                "take care",
            ),
        )

        if (
            previous_is_farewell
            and not following_is_farewell
        ):
            score -= 2.0




        if previous == following:
            score -= 1.0

        return score