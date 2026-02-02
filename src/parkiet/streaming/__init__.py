"""Streaming TTS module for real-time audio generation."""

from .config import StreamingConfig
from .dac_decoder import IncrementalDACDecoder
from .generation_state import StreamingGenerationState, StreamingChunk, TextUpdate
from .generator import StreamingTTSGenerator
from .text_updater import TextUpdateHandler
from .buffer_manager import StreamingBufferManager
from .session_manager import TTSSession, TTSSessionManager

__all__ = [
    "StreamingConfig",
    "IncrementalDACDecoder",
    "StreamingGenerationState",
    "StreamingChunk",
    "TextUpdate",
    "StreamingTTSGenerator",
    "TextUpdateHandler",
    "StreamingBufferManager",
    "TTSSession",
    "TTSSessionManager",
]


def create_app(model=None, config: StreamingConfig | None = None):
    """
    Create the FastAPI application for streaming TTS.

    Requires the 'streaming' extra: uv sync --extra streaming

    Args:
        model: The Dia TTS model. If None, will be loaded on startup.
        config: Streaming configuration.

    Returns:
        FastAPI application instance.
    """
    from .server import create_app as _create_app

    return _create_app(model, config)


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    tcp_port: int = 8001,
    model=None,
    config: StreamingConfig | None = None,
):
    """
    Run the streaming TTS server.

    Requires the 'streaming' extra: uv sync --extra streaming

    Args:
        host: Host to bind to.
        port: Port for WebSocket server.
        tcp_port: Port for TCP server.
        model: Pre-loaded Dia model (optional).
        config: Streaming configuration.
    """
    from .server import run_server as _run_server

    _run_server(host, port, tcp_port, model, config)
