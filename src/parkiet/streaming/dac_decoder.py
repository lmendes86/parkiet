"""Incremental DAC decoder for streaming audio generation."""

from collections import deque

import numpy as np
import torch

from .config import StreamingConfig


class IncrementalDACDecoder:
    """
    Decodes DAC frames incrementally as they're generated.

    The DAC codec uses a delay pattern where different channels have different
    temporal offsets. This means we need to buffer frames before we can decode
    them. The minimum buffer size is max(delay_pattern) + 1 frames.

    Example with delay_pattern = (0, 8, 9, 10, 11, 12, 13, 14, 15):
    - Channel 0 has delay 0: Can be decoded immediately
    - Channel 8 has delay 15: Need to wait 15 more frames

    To decode frame at position T, we need frames from T to T+max_delay.
    """

    def __init__(
        self,
        dac_model,
        config: StreamingConfig,
        device: torch.device | None = None,
    ):
        """
        Initialize the incremental DAC decoder.

        Args:
            dac_model: The DAC model for decoding codes to audio.
            config: Streaming configuration.
            device: Device to use for tensors.
        """
        self.dac_model = dac_model
        self.config = config
        self.delay_pattern = config.delay_pattern
        self.max_delay = config.max_delay
        self.num_channels = config.num_channels
        self.device = device or (
            next(dac_model.parameters()).device if dac_model else torch.device("cpu")
        )

        # Frame buffer - stores frames as they arrive
        self.frame_buffer: deque[torch.Tensor] = deque()

        # Position tracking
        self.total_frames_received = 0
        self.decoded_position = 0  # Number of frames successfully decoded

        # EOS handling
        self.eos_received = False
        self.eos_position: int | None = None

    def reset(self) -> None:
        """Reset decoder state for a new generation."""
        self.frame_buffer.clear()
        self.total_frames_received = 0
        self.decoded_position = 0
        self.eos_received = False
        self.eos_position = None

    def add_frame(self, frame: torch.Tensor) -> np.ndarray | None:
        """
        Add a generated frame (9 channels) to the buffer.

        Args:
            frame: Shape [num_channels] - one timestep of DAC codes.

        Returns:
            Decoded audio samples if enough frames are buffered, else None.
        """
        # Ensure frame is on correct device and has right shape
        if frame.dim() == 1:
            frame = frame.unsqueeze(0)  # [C] -> [1, C]
        frame = frame.to(self.device)

        self.frame_buffer.append(frame.squeeze(0))  # Store as [C]
        self.total_frames_received += 1

        # Check for EOS token in channel 0
        eos_token_id = 1024  # From DiaConfig
        if not self.eos_received and frame[0, 0].item() == eos_token_id:
            self.eos_received = True
            self.eos_position = self.total_frames_received - 1

        # Try to decode
        return self._try_decode()

    def _try_decode(self) -> np.ndarray | None:
        """
        Try to decode buffered frames.

        Returns:
            Decoded audio if frames are available, else None.
        """
        buffer_size = len(self.frame_buffer)

        # Calculate how many frames we can decode
        # We need max_delay additional frames after each frame to decode it
        available_for_decode = buffer_size - self.max_delay

        if available_for_decode <= self.decoded_position:
            # Not enough frames buffered yet
            return None

        # Decode in batches for efficiency
        frames_to_decode = min(
            available_for_decode - self.decoded_position,
            self.config.dac_decode_batch_size,
        )

        if frames_to_decode <= 0:
            return None

        # Get the segment to decode
        start_idx = self.decoded_position
        end_idx = start_idx + frames_to_decode

        # We need frames from start_idx to end_idx + max_delay for reverting delay
        segment_end = end_idx + self.max_delay
        segment_frames = [self.frame_buffer[i] for i in range(start_idx, segment_end)]
        codes_segment = torch.stack(segment_frames, dim=0)  # [T_segment, C]

        # Revert delay pattern for this segment
        reverted = self._revert_delay_segment(codes_segment, frames_to_decode)

        # Decode to audio
        audio = self._decode_segment(reverted)

        # Update position
        self.decoded_position = end_idx

        # Optionally trim buffer to prevent unbounded growth
        # Keep some margin for safety
        if self.decoded_position > self.max_delay * 2:
            trim_count = self.decoded_position - self.max_delay
            for _ in range(trim_count):
                self.frame_buffer.popleft()
            self.decoded_position -= trim_count

        return audio

    def _revert_delay_segment(
        self, codes: torch.Tensor, num_frames: int
    ) -> torch.Tensor:
        """
        Revert delay pattern for a segment of codes.

        Args:
            codes: Shape [T_segment, C] where T_segment >= num_frames + max_delay
            num_frames: Number of output frames to produce

        Returns:
            Reverted codes of shape [num_frames, C]
        """
        reverted = torch.zeros(
            num_frames, self.num_channels, dtype=codes.dtype, device=codes.device
        )

        for c, delay in enumerate(self.delay_pattern):
            # For output frame t, we need input frame t + delay
            reverted[:, c] = codes[delay : delay + num_frames, c]

        return reverted

    @torch.inference_mode()
    def _decode_segment(self, codes: torch.Tensor) -> np.ndarray:
        """
        Decode a segment of reverted codes to audio.

        Args:
            codes: Shape [T, C] - reverted (non-delayed) codes

        Returns:
            Audio samples as numpy array
        """
        # Validate codes: set invalid values (EOS=1024, PAD=1025, BOS=1026, negatives) to 0
        # Valid DAC codes are 0-1023
        invalid_mask = (codes < 0) | (codes > 1023)
        codes = codes.clone()
        codes[invalid_mask] = 0

        # DAC expects [B, C, T] format
        codes = codes.unsqueeze(0).transpose(1, 2)  # [1, C, T]

        # Decode through DAC
        audio_values, _, _ = self.dac_model.quantizer.from_codes(codes)
        audio_values = self.dac_model.decode(audio_values)

        return audio_values.squeeze().cpu().numpy()

    def flush(self) -> np.ndarray | None:
        """
        Flush remaining buffered frames.

        Call this when generation is complete (EOS received) to get
        any remaining audio in the buffer.

        Returns:
            Remaining audio if any, else None.
        """
        if len(self.frame_buffer) == 0:
            return None

        # For final flush, we can decode everything remaining
        # Pad with zeros if needed for delay pattern
        buffer_size = len(self.frame_buffer)
        remaining = buffer_size - self.decoded_position

        if remaining <= 0:
            return None

        # Stack all remaining frames
        segment_frames = [
            self.frame_buffer[i]
            for i in range(self.decoded_position, buffer_size)
        ]

        # Pad with zeros for the delay pattern
        pad_frames = self.max_delay
        zero_frame = torch.zeros(
            self.num_channels, dtype=segment_frames[0].dtype, device=self.device
        )
        segment_frames.extend([zero_frame] * pad_frames)

        codes_segment = torch.stack(segment_frames, dim=0)

        # Revert delay pattern
        reverted = self._revert_delay_segment(codes_segment, remaining)

        # Decode to audio
        audio = self._decode_segment(reverted)

        self.decoded_position = buffer_size

        return audio

    @property
    def buffer_level(self) -> int:
        """Number of frames currently buffered."""
        return len(self.frame_buffer) - self.decoded_position

    @property
    def can_decode(self) -> bool:
        """Whether there are frames available for decoding."""
        return self.buffer_level > self.max_delay

    @property
    def is_complete(self) -> bool:
        """Whether EOS has been received and all frames decoded."""
        return self.eos_received and self.buffer_level <= self.max_delay
