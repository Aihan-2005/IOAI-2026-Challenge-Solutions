from __future__ import annotations

from typing import Protocol

from src.domain import Dialogue
from src.order_utils import order_to_rank


class RankingPredictor(Protocol):
    """
    Interface implemented by every Task 1 solution.
    """

    name: str

    def predict(
        self,
        dialogue: Dialogue,
    ) -> list[int]:
        """
        Return IOAI rank representation for one dialogue.
        """
        ...


class PrefixIndexBaseline:
    """
    Reimplementation of the official deterministic baseline.

    Known prefix chunks are placed first.
    All remaining chunks stay in shuffled index order.
    """

    name = "prefix-index-baseline"

    def predict(
        self,
        dialogue: Dialogue,
    ) -> list[int]:
        first, second = dialogue.prefix

        chronological_order = [
            first,
            second,
            *(
                index
                for index in range(dialogue.n_chunks)
                if index not in (first, second)
            ),
        ]

        return order_to_rank(chronological_order)
    