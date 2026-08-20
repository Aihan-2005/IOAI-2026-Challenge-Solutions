from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.domain import Dialogue
from src.transcription import (
    ChunkTranscript,
    DialogueTranscript,
)


@dataclass(
    frozen=True,
    slots=True,
)
class WhisperConfig:
    model_id: str = (
        "openai/whisper-small"
    )

    sampling_rate: int = 16_000

    language: str = "en"

    task: str = "transcribe"

    max_new_tokens: int = 128

    device: str | None = None


class WhisperTranscriber:
    """
    GPU-backed Whisper transcription backend.

    Intended for Colab / remote NVIDIA GPU execution.

    The class satisfies the project's Transcriber interface
    without coupling the rest of the codebase to PyTorch or
    Transformers.
    """

    name = "whisper-small"

    def __init__(
        self,
        config: WhisperConfig | None = None,
    ) -> None:
        self.config = (
            config
            or WhisperConfig()
        )

        self._initialize_backend()

    def _initialize_backend(
        self,
    ) -> None:
        import torch

        from transformers import (
            AutoModelForSpeechSeq2Seq,
            AutoProcessor,
        )

        if self.config.device:
            device = torch.device(
                self.config.device
            )
        elif torch.cuda.is_available():
            device = torch.device(
                "cuda"
            )
        else:
            device = torch.device(
                "cpu"
            )

        self._torch = torch
        self._device = device

        self._dtype = (
            torch.float16
            if device.type == "cuda"
            else torch.float32
        )

        self._processor = (
            AutoProcessor.from_pretrained(
                self.config.model_id
            )
        )

        self._model = (
            AutoModelForSpeechSeq2Seq
            .from_pretrained(
                self.config.model_id,
                torch_dtype=self._dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
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

    @property
    def dtype(self) -> str:
        return str(
            self._dtype
        )

    def _load_audio(
        self,
        path: str,
    ) -> Any:
        import librosa

        audio, _ = librosa.load(
            path,
            sr=self.config.sampling_rate,
            mono=True,
        )

        return audio

    def transcribe_file(
        self,
        path: str,
    ) -> str:
        audio = self._load_audio(
            path
        )

        inputs = (
            self._processor(
                audio,
                sampling_rate=(
                    self.config.sampling_rate
                ),
                return_tensors="pt",
            )
        )

        input_features = (
            inputs.input_features.to(
                device=self._device,
                dtype=self._dtype,
            )
        )

        with self._torch.inference_mode():
            generated_ids = (
                self._model.generate(
                    input_features,
                    language=(
                        self.config.language
                    ),
                    task=self.config.task,
                    max_new_tokens=(
                        self.config
                        .max_new_tokens
                    ),
                )
            )

        text = (
            self._processor
            .batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )[0]
            .strip()
        )

        return text

    def transcribe(
        self,
        dialogue: Dialogue,
    ) -> DialogueTranscript:
        chunks: list[
            ChunkTranscript
        ] = []

        for chunk_index, path in enumerate(
            dialogue.chunk_paths
        ):
            text = self.transcribe_file(
                str(path)
            )

            chunks.append(
                ChunkTranscript(
                    chunk_index=(
                        chunk_index
                    ),
                    text=text,
                )
            )

        return DialogueTranscript(
            dialogue_id=(
                dialogue.dialogue_id
            ),
            chunks=tuple(chunks),
        )