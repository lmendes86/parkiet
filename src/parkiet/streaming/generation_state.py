"""State management for streaming TTS generation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch


class GenerationStatus(str, Enum):
    """Status of the generation process."""

    INITIALIZING = "initializing"
    GENERATING = "generating"
    PAUSED = "paused"
    COMPLETING = "completing"  # EOS detected, draining buffer
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class StreamingChunk:
    """A chunk of streamed audio data."""

    audio_data: np.ndarray
    """PCM audio samples (float32, 44.1kHz)."""

    timestamp_ms: float
    """Timestamp in milliseconds from start of generation."""

    frame_index: int
    """Frame index in the generation sequence."""

    is_final: bool = False
    """Whether this is the final chunk."""

    def to_bytes(self, dtype: str = "float32") -> bytes:
        """
        Convert audio data to bytes.

        Args:
            dtype: Output dtype - 'float32' or 'int16'

        Returns:
            Raw PCM bytes
        """
        if dtype == "float32":
            return self.audio_data.astype(np.float32).tobytes()
        elif dtype == "int16":
            # Convert float32 [-1, 1] to int16 [-32768, 32767]
            int16_data = (self.audio_data * 32767).astype(np.int16)
            return int16_data.tobytes()
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")


@dataclass
class TextUpdate:
    """Request to update the text during generation."""

    new_text: str
    """The new text to generate."""

    strategy: str = "continue"
    """
    Update strategy:
    - 'continue': Keep decoder self-attention cache, update cross-attention only
    - 'restart': Discard all decoder state, start fresh
    """


@dataclass
class StreamingGenerationState:
    """
    Encapsulates all state for an ongoing streaming generation.

    This state object tracks:
    - Text encoding state (immutable after encoding)
    - Encoder output and cross-attention cache
    - Decoder state (mutable during generation)
    - Audio buffer state
    - Generation control state
    """

    # Generation ID
    generation_id: str

    # Text state (immutable after encoding)
    text: str
    text_tokens: torch.Tensor
    text_version: int = 0

    # Encoder state
    enc_state: Any = None  # EncoderInferenceState
    encoder_out: torch.Tensor | None = None
    cross_attn_cache: list | None = None  # list[KVCache]

    # Decoder state
    dec_state: Any = None  # DecoderInferenceState
    dec_output: Any = None  # DecoderOutput
    current_step: int = 0

    # Prefill state
    prefill_steps: list[int] = field(default_factory=list)

    # EOS handling
    eos_detected: bool = False
    eos_countdown: int = -1
    finished_step: int = -1

    # Generation status
    status: GenerationStatus = GenerationStatus.INITIALIZING

    # Audio tracking
    total_audio_samples: int = 0
    total_audio_duration_ms: float = 0.0

    # Pending text update
    pending_text_update: TextUpdate | None = None

    # Error information
    error_message: str | None = None

    # Device
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    @property
    def is_active(self) -> bool:
        """Whether generation is actively producing tokens."""
        return self.status in (
            GenerationStatus.GENERATING,
            GenerationStatus.COMPLETING,
        )

    @property
    def is_complete(self) -> bool:
        """Whether generation is complete."""
        return self.status == GenerationStatus.COMPLETE

    @property
    def has_error(self) -> bool:
        """Whether an error occurred."""
        return self.status == GenerationStatus.ERROR

    def mark_eos_detected(self, step: int, max_delay: int) -> None:
        """Mark that EOS has been detected."""
        if not self.eos_detected:
            self.eos_detected = True
            self.eos_countdown = max_delay
            self.finished_step = step
            self.status = GenerationStatus.COMPLETING

    def decrement_eos_countdown(self) -> bool:
        """
        Decrement EOS countdown.

        Returns:
            True if generation should continue, False if complete.
        """
        if self.eos_countdown > 0:
            self.eos_countdown -= 1
            return True
        elif self.eos_countdown == 0:
            self.status = GenerationStatus.COMPLETE
            return False
        return True  # Not in EOS countdown

    def set_error(self, message: str) -> None:
        """Set error state."""
        self.status = GenerationStatus.ERROR
        self.error_message = message

    def update_audio_stats(self, num_samples: int, sample_rate: int = 44100) -> None:
        """Update audio statistics."""
        self.total_audio_samples += num_samples
        self.total_audio_duration_ms = self.total_audio_samples / sample_rate * 1000
