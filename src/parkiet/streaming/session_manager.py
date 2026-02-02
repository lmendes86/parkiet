"""Session management for streaming TTS."""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .config import StreamingConfig
from .generator import StreamingTTSGenerator
from .generation_state import TextUpdate
from .buffer_manager import StreamingBufferManager

logger = logging.getLogger(__name__)


class SessionState(str, Enum):
    """State of a TTS session."""

    IDLE = "idle"
    GENERATING = "generating"
    PAUSED = "paused"
    COMPLETE = "complete"
    ERROR = "error"
    CLOSED = "closed"


@dataclass
class TTSSession:
    """
    Represents an active TTS streaming session.

    A session manages the lifecycle of a TTS generation, including:
    - Starting/stopping generation
    - Text updates during generation
    - Audio output buffering
    - State tracking
    """

    session_id: str
    model: object  # Dia model
    config: StreamingConfig

    # Internal state
    state: SessionState = SessionState.IDLE
    generator: StreamingTTSGenerator | None = None
    buffer_manager: StreamingBufferManager | None = None

    # Generation task
    _generation_task: asyncio.Task | None = field(default=None, repr=False)

    # Queues for communication
    _audio_queue: asyncio.Queue | None = field(default=None, repr=False)
    _text_update_queue: asyncio.Queue | None = field(default=None, repr=False)

    # Error tracking
    error_message: str | None = None

    def __post_init__(self):
        """Initialize queues and generator."""
        self._audio_queue = asyncio.Queue(maxsize=100)
        self._text_update_queue = asyncio.Queue()
        self.generator = StreamingTTSGenerator(self.model, self.config)
        self.buffer_manager = StreamingBufferManager(self.config)

    async def start_generation(
        self,
        text: str,
        cfg_scale: float | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        **kwargs,
    ) -> None:
        """
        Start generating audio for the given text.

        Args:
            text: Text to synthesize.
            cfg_scale: Classifier-free guidance scale.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            **kwargs: Additional generation parameters.
        """
        if self.state not in (SessionState.IDLE, SessionState.COMPLETE):
            raise RuntimeError(
                f"Cannot start generation in state {self.state}"
            )

        self.state = SessionState.GENERATING
        self.buffer_manager.reset()

        # Start generation in background task
        self._generation_task = asyncio.create_task(
            self._generation_loop(
                text=text,
                cfg_scale=cfg_scale,
                temperature=temperature,
                top_p=top_p,
                **kwargs,
            )
        )

    async def _generation_loop(
        self,
        text: str,
        **kwargs,
    ) -> None:
        """Main generation loop running in background."""
        try:
            gen = self.generator.generate_stream(text, **kwargs)

            async for chunk in gen:
                if self.state == SessionState.CLOSED:
                    break

                # Add to buffer
                self.buffer_manager.add_audio(chunk.audio_data)

                # Also queue for direct access
                try:
                    self._audio_queue.put_nowait(chunk)
                except asyncio.QueueFull:
                    # Drop oldest if queue is full
                    try:
                        self._audio_queue.get_nowait()
                        self._audio_queue.put_nowait(chunk)
                    except asyncio.QueueEmpty:
                        pass

                # Check for text updates
                try:
                    text_update = self._text_update_queue.get_nowait()
                    # Send update to generator
                    await gen.asend(text_update)
                except asyncio.QueueEmpty:
                    pass

            # Generation complete
            self.buffer_manager.signal_generation_complete()
            if self.state != SessionState.CLOSED:
                self.state = SessionState.COMPLETE

        except Exception as e:
            logger.exception(f"Error in generation loop: {e}")
            self.state = SessionState.ERROR
            self.error_message = str(e)
            raise

    async def update_text(
        self,
        new_text: str,
        strategy: str = "continue",
    ) -> None:
        """
        Update the text being generated.

        Args:
            new_text: New text to generate.
            strategy: Update strategy ('continue' or 'restart').
        """
        if self.state != SessionState.GENERATING:
            raise RuntimeError(
                f"Cannot update text in state {self.state}"
            )

        update = TextUpdate(new_text=new_text, strategy=strategy)
        await self._text_update_queue.put(update)

    async def get_audio_chunk(
        self,
        duration_ms: float = 20.0,
        timeout: float | None = 1.0,
    ) -> np.ndarray | None:
        """
        Get the next audio chunk from the buffer.

        Args:
            duration_ms: Desired chunk duration in milliseconds.
            timeout: Maximum time to wait.

        Returns:
            Audio samples or None if complete.
        """
        return await self.buffer_manager.get_chunk(
            duration_ms=duration_ms,
            timeout=timeout,
        )

    async def get_raw_chunk(
        self,
        timeout: float = 1.0,
    ):
        """
        Get the next raw StreamingChunk from the queue.

        Args:
            timeout: Maximum time to wait.

        Returns:
            StreamingChunk or None if timeout/complete.
        """
        try:
            return await asyncio.wait_for(
                self._audio_queue.get(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return None

    async def stop(self) -> None:
        """Stop the current generation."""
        if self._generation_task and not self._generation_task.done():
            self._generation_task.cancel()
            try:
                await self._generation_task
            except asyncio.CancelledError:
                pass

        self.state = SessionState.IDLE

    async def close(self) -> None:
        """Close the session and cleanup resources."""
        self.state = SessionState.CLOSED

        if self._generation_task and not self._generation_task.done():
            self._generation_task.cancel()
            try:
                await self._generation_task
            except asyncio.CancelledError:
                pass

        # Clear queues
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    @property
    def is_active(self) -> bool:
        """Whether the session is actively generating."""
        return self.state == SessionState.GENERATING

    @property
    def is_complete(self) -> bool:
        """Whether generation is complete."""
        return self.state == SessionState.COMPLETE

    @property
    def has_error(self) -> bool:
        """Whether an error occurred."""
        return self.state == SessionState.ERROR


class TTSSessionManager:
    """
    Manages multiple concurrent TTS sessions.

    This class handles:
    - Creating and destroying sessions
    - Enforcing session limits
    - Session lookup and validation
    """

    def __init__(
        self,
        model,  # Dia model
        config: StreamingConfig | None = None,
    ):
        """
        Initialize the session manager.

        Args:
            model: The Dia TTS model (shared across all sessions).
            config: Streaming configuration.
        """
        self.model = model
        self.config = config or StreamingConfig()
        self._sessions: dict[str, TTSSession] = {}
        self._lock = asyncio.Lock()

    async def create_session(
        self,
        session_id: str | None = None,
    ) -> TTSSession:
        """
        Create a new TTS session.

        Args:
            session_id: Optional session ID. Generated if not provided.

        Returns:
            The created session.

        Raises:
            RuntimeError: If maximum sessions reached.
        """
        async with self._lock:
            if len(self._sessions) >= self.config.max_sessions:
                raise RuntimeError(
                    f"Maximum sessions ({self.config.max_sessions}) reached"
                )

            session_id = session_id or str(uuid.uuid4())

            if session_id in self._sessions:
                raise ValueError(f"Session {session_id} already exists")

            session = TTSSession(
                session_id=session_id,
                model=self.model,
                config=self.config,
            )

            self._sessions[session_id] = session
            logger.info(f"Created session {session_id}")

            return session

    async def get_session(self, session_id: str) -> TTSSession | None:
        """
        Get a session by ID.

        Args:
            session_id: The session ID.

        Returns:
            The session or None if not found.
        """
        return self._sessions.get(session_id)

    async def close_session(self, session_id: str) -> None:
        """
        Close and remove a session.

        Args:
            session_id: The session ID to close.
        """
        async with self._lock:
            session = self._sessions.pop(session_id, None)
            if session:
                await session.close()
                logger.info(f"Closed session {session_id}")

    async def close_all(self) -> None:
        """Close all sessions."""
        async with self._lock:
            for session_id, session in list(self._sessions.items()):
                await session.close()
            self._sessions.clear()
            logger.info("Closed all sessions")

    @property
    def active_sessions(self) -> int:
        """Number of active sessions."""
        return len(self._sessions)

    @property
    def session_ids(self) -> list[str]:
        """List of active session IDs."""
        return list(self._sessions.keys())
