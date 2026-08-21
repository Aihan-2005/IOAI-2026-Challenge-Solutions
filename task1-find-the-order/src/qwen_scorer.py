from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.transition import (
    TransitionPair,
)


@dataclass(
    frozen=True,
    slots=True,
)
class QwenScorerConfig:
    model_id: str = (
        "Qwen/Qwen2.5-0.5B"
    )

    batch_size: int = 16

    max_prompt_tokens: int = 384

    max_candidate_tokens: int = 128

    device: str | None = None


class QwenContextualLiftScorer:
    """
    Scores whether candidate B is a plausible continuation of A.

    score(A, B) =

        avg_log P(B | A)
        -
        avg_log P(B | neutral context)

    This contrastive formulation reduces generic-fluency and
    candidate-length bias.
    """

    name = (
        "qwen2.5-0.5b-contextual-lift-v1"
    )

    def __init__(
        self,
        config: QwenScorerConfig | None = None,
    ) -> None:
        self.config = (
            config
            or QwenScorerConfig()
        )

        self._initialize()

    def _initialize(
        self,
    ) -> None:
        import torch

        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        self._torch = torch

        if self.config.device:
            self._device = torch.device(
                self.config.device
            )

        elif torch.cuda.is_available():
            self._device = torch.device(
                "cuda"
            )

        else:
            self._device = torch.device(
                "cpu"
            )

        self._dtype = (
            torch.float16
            if self._device.type == "cuda"
            else torch.float32
        )

        self._tokenizer = (
            AutoTokenizer.from_pretrained(
                self.config.model_id,
                use_fast=True,
            )
        )

        if (
            self._tokenizer.pad_token_id
            is None
        ):
            self._tokenizer.pad_token = (
                self._tokenizer.eos_token
            )

        self._model = (
            AutoModelForCausalLM
            .from_pretrained(
                self.config.model_id,
                torch_dtype=self._dtype,
                low_cpu_mem_usage=True,
            )
        )

        self._model.to(
            self._device
        )

        self._model.eval()

    @property
    def device(self) -> str:
        return str(
            self._device
        )

    def _encode_prompt(
        self,
        text: str,
    ) -> list[int]:
        ids = self._tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        ids = ids[
            -self.config.max_prompt_tokens:
        ]

        if ids:
            return ids

        fallback = (
            self._tokenizer.eos_token_id
        )

        if fallback is None:
            raise RuntimeError(
                "Tokenizer has no fallback token"
            )

        return [
            fallback
        ]

    def _encode_candidate(
        self,
        text: str,
    ) -> list[int]:
        ids = self._tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        return ids[
            : self.config.max_candidate_tokens
        ]

    def _average_continuation_logprob(
        self,
        *,
        prompts: Sequence[str],
        candidates: Sequence[str],
    ) -> list[float]:
        if len(prompts) != len(candidates):
            raise ValueError(
                "prompts/candidates length mismatch"
            )

        results: list[float] = []

        for batch_start in range(
            0,
            len(prompts),
            self.config.batch_size,
        ):
            batch_prompts = prompts[
                batch_start:
                batch_start
                + self.config.batch_size
            ]

            batch_candidates = candidates[
                batch_start:
                batch_start
                + self.config.batch_size
            ]

            encoded: list[
                tuple[
                    list[int],
                    list[int],
                ]
            ] = []

            for prompt, candidate in zip(
                batch_prompts,
                batch_candidates,
                strict=True,
            ):
                prompt_ids = (
                    self._encode_prompt(
                        prompt
                    )
                )

                candidate_ids = (
                    self._encode_candidate(
                        candidate
                    )
                )

                encoded.append(
                    (
                        prompt_ids,
                        candidate_ids,
                    )
                )

            max_length = max(
                len(prompt_ids)
                + len(candidate_ids)
                for (
                    prompt_ids,
                    candidate_ids,
                )
                in encoded
            )

            pad_id = (
                self._tokenizer.pad_token_id
            )

            input_rows: list[
                list[int]
            ] = []

            attention_rows: list[
                list[int]
            ] = []

            for (
                prompt_ids,
                candidate_ids,
            ) in encoded:
                sequence = (
                    prompt_ids
                    + candidate_ids
                )

                padding = (
                    max_length
                    - len(sequence)
                )

                input_rows.append(
                    sequence
                    + [pad_id] * padding
                )

                attention_rows.append(
                    [1] * len(sequence)
                    + [0] * padding
                )

            input_ids = (
                self._torch.tensor(
                    input_rows,
                    dtype=self._torch.long,
                    device=self._device,
                )
            )

            attention_mask = (
                self._torch.tensor(
                    attention_rows,
                    dtype=self._torch.long,
                    device=self._device,
                )
            )

            with (
                self._torch.inference_mode()
            ):
                logits = self._model(
                    input_ids=input_ids,
                    attention_mask=(
                        attention_mask
                    ),
                ).logits

            for row_index, (
                prompt_ids,
                candidate_ids,
            ) in enumerate(
                encoded
            ):
                if not candidate_ids:
                    results.append(
                        -100.0
                    )

                    continue

                start = len(
                    prompt_ids
                )

                end = (
                    start
                    + len(candidate_ids)
                )

                relevant_logits = (
                    logits[
                        row_index,
                        start - 1:
                        end - 1,
                        :,
                    ]
                ).float()

                target_ids = (
                    self._torch.tensor(
                        candidate_ids,
                        dtype=self._torch.long,
                        device=self._device,
                    )
                )

                token_log_probs = (
                    self._torch
                    .log_softmax(
                        relevant_logits,
                        dim=-1,
                    )
                    .gather(
                        dim=-1,
                        index=(
                            target_ids
                            .unsqueeze(-1)
                        ),
                    )
                    .squeeze(-1)
                )

                results.append(
                    float(
                        token_log_probs
                        .mean()
                        .item()
                    )
                )

        return results

    @staticmethod
    def _conditional_prompt(
        previous_text: str,
    ) -> str:
        return (
            "Previous dialogue turn:\n"
            f"{previous_text.strip()}\n\n"
            "Next dialogue turn:\n"
        )

    @staticmethod
    def _neutral_prompt() -> str:
        return (
            "Next dialogue turn:\n"
        )

    def score_many(
        self,
        pairs: Sequence[TransitionPair],
    ) -> list[float]:
        if not pairs:
            return []

        candidates = [
            following.strip()
            for _, following
            in pairs
        ]

        conditional_prompts = [
            self._conditional_prompt(
                previous
            )
            for previous, _
            in pairs
        ]

        neutral_prompt = (
            self._neutral_prompt()
        )

        neutral_prompts = [
            neutral_prompt
            for _ in pairs
        ]

        conditional_scores = (
            self._average_continuation_logprob(
                prompts=(
                    conditional_prompts
                ),
                candidates=candidates,
            )
        )

        neutral_scores = (
            self._average_continuation_logprob(
                prompts=neutral_prompts,
                candidates=candidates,
            )
        )

        return [
            conditional
            - neutral
            for (
                conditional,
                neutral,
            ) in zip(
                conditional_scores,
                neutral_scores,
                strict=True,
            )
        ]

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