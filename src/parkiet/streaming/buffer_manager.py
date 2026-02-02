"""Buffer management for smooth audio streaming."""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .config import StreamingConfig


class BufferState(Enum):
    """State of the streaming buffer."""

    FILLING = "filling"  # Initial buffer fill phase
    STREAMING = "streaming"  # Normal streaming operation
    DRAINING = "draining"  # Generation complete, draining buffer
    EMPTY = "empty"  # Buffer is empty


@dataclass
class BufferStats:
    """Statistics about the buffer state."""

    state: BufferState
    pcm_samples: int
    pcm_duration_ms: float
    chunks_buffered: int
    total_samples_output: int
    total_chunks_output: int


class StreamingBufferManager:
    """
    Manages buffering for smooth audio streaming.

    This buffer sits between the generator (which produces audio in bursts)
    and the output (which needs steady chunks). It smooths out variations
    in generation speed.

    Features:
    - Configurable minimum buffer before first output
    - Maximum buffer limit to bound latency
    - Async interface for integration with event loops
    """

    def __init__(self, config: StreamingConfig):
        """
        Initialize the buffer manager.

        Args:
            config: Streaming configuration.
        """
        self.config = config
        self.sample_rate = config.sample_rate

        # Buffer state
        self.state = BufferState.FILLING
        self._buffer: deque[np.ndarray] = deque()
        self._total_samples = 0

        # Output tracking
        self._samples_output = 0
        self._chunks_output = 0

        # Async event for notifying waiters
        self._data_available = asyncio.Event()
        self._generation_complete = False

    def reset(self) -> None:
        """Reset buffer to initial state."""
        self.state = BufferState.FILLING
        self._buffer.clear()
        self._total_samples = 0
        self._samples_output = 0
        self._chunks_output = 0
        self._data_available.clear()
        self._generation_complete = False

    def add_audio(self, audio: np.ndarray) -> None:
        """
        Add audio samples to the buffer.

        Args:
            audio: Audio samples to buffer (float32, 44.1kHz).
        """
        if len(audio) == 0:
            return

        self._buffer.append(audio)
        self._total_samples += len(audio)

        # Check buffer limits
        self._enforce_max_buffer()

        # Update state
        self._update_state()

        # Signal that data is available
        self._data_available.set()

    def _enforce_max_buffer(self) -> None:
        """Enforce maximum buffer size by dropping oldest samples."""
        max_samples = int(self.config.output_max_ms * self.sample_rate / 1000)

        while self._total_samples > max_samples and self._buffer:
            oldest = self._buffer.popleft()
            self._total_samples -= len(oldest)

    def _update_state(self) -> None:
        """Update buffer state based on current fill level."""
        if self._generation_complete:
            if self._total_samples == 0:
                self.state = BufferState.EMPTY
            else:
                self.state = BufferState.DRAINING
        elif self.state == BufferState.FILLING:
            min_samples = int(self.config.output_min_ms * self.sample_rate / 1000)
            if self._total_samples >= min_samples:
                self.state = BufferState.STREAMING

    def signal_generation_complete(self) -> None:
        """Signal that generation is complete."""
        self._generation_complete = True
        self._update_state()
        self._data_available.set()

    async def get_chunk(
        self,
        duration_ms: float = 20.0,
        timeout: float | None = 1.0,
    ) -> np.ndarray | None:
        """
        Get a chunk of audio from the buffer.

        Args:
            duration_ms: Desired chunk duration in milliseconds.
            timeout: Maximum time to wait for data (None for no timeout).

        Returns:
            Audio chunk as numpy array, or None if buffer is empty
            and generation is complete.
        """
        chunk_samples = int(duration_ms * self.sample_rate / 1000)

        while True:
            # Check if we have enough data
            if self._can_yield(chunk_samples):
                return self._extract_chunk(chunk_samples)

            # Check if we're done
            if self.state == BufferState.EMPTY:
                return None

            # If draining, return whatever we have
            if self.state == BufferState.DRAINING and self._total_samples > 0:
                return self._extract_all()

            # Wait for more data
            self._data_available.clear()
            try:
                if timeout is not None:
                    await asyncio.wait_for(
                        self._data_available.wait(),
                        timeout=timeout,
                    )
                else:
                    await self._data_available.wait()
            except asyncio.TimeoutError:
                # Return partial chunk if we have any data
                if self._total_samples > 0:
                    return self._extract_all()
                return None

    def _can_yield(self, chunk_samples: int) -> bool:
        """Check if we can yield a chunk of the requested size."""
        if self.state == BufferState.FILLING:
            return False
        return self._total_samples >= chunk_samples

    def _extract_chunk(self, num_samples: int) -> np.ndarray:
        """Extract exactly num_samples from the buffer."""
        chunks = []
        remaining = num_samples

        while remaining > 0 and self._buffer:
            chunk = self._buffer[0]

            if len(chunk) <= remaining:
                # Use entire chunk
                chunks.append(chunk)
                remaining -= len(chunk)
                self._buffer.popleft()
                self._total_samples -= len(chunk)
            else:
                # Split chunk
                chunks.append(chunk[:remaining])
                self._buffer[0] = chunk[remaining:]
                self._total_samples -= remaining
                remaining = 0

        result = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
        self._samples_output += len(result)
        self._chunks_output += 1

        self._update_state()
        return result

    def _extract_all(self) -> np.ndarray:
        """Extract all remaining samples from the buffer."""
        if not self._buffer:
            return np.array([], dtype=np.float32)

        chunks = list(self._buffer)
        self._buffer.clear()
        self._total_samples = 0

        result = np.concatenate(chunks)
        self._samples_output += len(result)
        self._chunks_output += 1

        self._update_state()
        return result

    @property
    def buffer_duration_ms(self) -> float:
        """Current buffer duration in milliseconds."""
        return self._total_samples / self.sample_rate * 1000

    @property
    def is_ready(self) -> bool:
        """Whether buffer is ready to start streaming."""
        return self.state in (BufferState.STREAMING, BufferState.DRAINING)

    @property
    def is_complete(self) -> bool:
        """Whether buffer is fully drained."""
        return self.state == BufferState.EMPTY

    def get_stats(self) -> BufferStats:
        """Get current buffer statistics."""
        return BufferStats(
            state=self.state,
            pcm_samples=self._total_samples,
            pcm_duration_ms=self.buffer_duration_ms,
            chunks_buffered=len(self._buffer),
            total_samples_output=self._samples_output,
            total_chunks_output=self._chunks_output,
        )
