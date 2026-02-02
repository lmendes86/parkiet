"""Configuration for streaming TTS."""

from dataclasses import dataclass, field


@dataclass
class StreamingConfig:
    """Configuration for streaming TTS generation."""

    # DAC decoder buffer settings
    dac_min_frames: int = 20
    """Minimum frames before first decode (~232ms). Must be >= max(delay_pattern) + 1."""

    dac_decode_batch_size: int = 10
    """Number of frames to decode in each batch."""

    # Audio output buffer settings
    output_min_ms: int = 200
    """Minimum buffered audio (ms) before first output."""

    output_target_ms: int = 500
    """Target buffer level (ms) for smooth streaming."""

    output_max_ms: int = 2000
    """Maximum buffer (ms) to limit latency."""

    # Generation settings
    default_cfg_scale: float = 3.0
    """Default classifier-free guidance scale."""

    default_temperature: float = 1.2
    """Default sampling temperature."""

    default_top_p: float = 0.95
    """Default nucleus sampling threshold."""

    default_cfg_filter_top_k: int = 45
    """Default top-k for CFG filtering."""

    max_tokens: int = 3072
    """Maximum generation length in tokens."""

    # Server settings
    max_sessions: int = 10
    """Maximum concurrent TTS sessions."""

    websocket_port: int = 8000
    """Port for WebSocket server."""

    tcp_port: int = 8001
    """Port for raw TCP streaming server."""

    # Audio format
    sample_rate: int = 44100
    """Output sample rate (native DAC rate)."""

    samples_per_frame: int = 512
    """Samples per DAC frame."""

    num_channels: int = 9
    """Number of DAC codebook channels."""

    # Delay pattern (from DiaConfig)
    delay_pattern: tuple[int, ...] = field(
        default_factory=lambda: (0, 8, 9, 10, 11, 12, 13, 14, 15)
    )
    """DAC delay pattern for multi-channel generation."""

    @property
    def max_delay(self) -> int:
        """Maximum delay in the delay pattern."""
        return max(self.delay_pattern)

    @property
    def frame_duration_ms(self) -> float:
        """Duration of one DAC frame in milliseconds."""
        return self.samples_per_frame / self.sample_rate * 1000

    @property
    def min_buffer_latency_ms(self) -> float:
        """Minimum latency due to delay pattern buffering."""
        return (self.max_delay + 1) * self.frame_duration_ms

    def validate(self) -> None:
        """Validate configuration settings."""
        if self.dac_min_frames < self.max_delay + 1:
            raise ValueError(
                f"dac_min_frames ({self.dac_min_frames}) must be >= "
                f"max_delay + 1 ({self.max_delay + 1})"
            )
        if self.output_min_ms < self.min_buffer_latency_ms:
            raise ValueError(
                f"output_min_ms ({self.output_min_ms}) should be >= "
                f"min_buffer_latency_ms ({self.min_buffer_latency_ms:.1f})"
            )
