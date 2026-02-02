"""FastAPI server for streaming TTS with WebSocket and TCP support."""

import asyncio
import base64
import json
import logging
from contextlib import asynccontextmanager

import numpy as np

from .config import StreamingConfig
from .session_manager import TTSSessionManager

logger = logging.getLogger(__name__)

# Global session manager (initialized on startup)
session_manager: TTSSessionManager | None = None


def create_app(
    model=None,
    config: StreamingConfig | None = None,
    model_path: str | None = None,
    use_hf_model: bool = False,
):
    """
    Create the FastAPI application.

    Args:
        model: The Dia TTS model. If None, will be loaded on startup.
        config: Streaming configuration.
        model_path: Path to local model checkpoint (.pth file).
        use_hf_model: If True, use HuggingFace Transformers model format.

    Returns:
        FastAPI application instance.
    """
    try:
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
        from fastapi.responses import JSONResponse, StreamingResponse
        from pydantic import BaseModel
    except ImportError:
        raise ImportError(
            "FastAPI is required for the streaming server. "
            "Install with: uv sync --extra streaming"
        )

    config = config or StreamingConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan handler."""
        global session_manager

        # Load model if not provided
        nonlocal model
        if model is None:
            logger.info("Loading Dia model...")
            from parkiet.dia.model import Dia

            if model_path:
                # Load from local checkpoint
                logger.info(f"Loading from local path: {model_path}")
                model = Dia.from_local(
                    config_path="config.json",
                    checkpoint_path=model_path,
                    compute_dtype="float32",
                )
            else:
                # Try to load from HuggingFace using the custom Dia class
                # Note: This requires the model to have safetensors/pytorch_model.bin format
                try:
                    model = Dia.from_pretrained("pevers/parkiet")
                except Exception as e:
                    logger.warning(f"Could not load with Dia.from_pretrained: {e}")
                    logger.info("Please provide a local model path with --model-path")
                    raise RuntimeError(
                        "Model loading failed. Use --model-path to specify a local .pth checkpoint. "
                        "Download from: https://huggingface.co/pevers/parkiet/resolve/main/dia-nl-v1.pth"
                    )
            logger.info("Model loaded successfully")

        # Create session manager
        session_manager = TTSSessionManager(model, config)
        logger.info(
            f"Session manager initialized (max_sessions={config.max_sessions})"
        )

        yield

        # Cleanup
        if session_manager:
            await session_manager.close_all()
            logger.info("Session manager closed")

    app = FastAPI(
        title="Parkiet Streaming TTS",
        description="Real-time streaming Text-to-Speech API for Dutch",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Pydantic models for request/response
    class TTSRequest(BaseModel):
        text: str
        cfg_scale: float = 3.0
        temperature: float = 1.2
        top_p: float = 0.95
        cfg_filter_top_k: int = 45

    class TTSResponse(BaseModel):
        session_id: str
        status: str
        message: str | None = None

    class SessionInfo(BaseModel):
        session_id: str
        state: str
        buffer_duration_ms: float
        error_message: str | None = None

    # REST endpoints

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "active_sessions": session_manager.active_sessions}

    @app.get("/sessions")
    async def list_sessions():
        """List active sessions."""
        return {"sessions": session_manager.session_ids}

    @app.post("/sessions", response_model=TTSResponse)
    async def create_session():
        """Create a new TTS session."""
        try:
            session = await session_manager.create_session()
            return TTSResponse(
                session_id=session.session_id,
                status="created",
            )
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str):
        """Delete a session."""
        session = await session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        await session_manager.close_session(session_id)
        return {"status": "deleted", "session_id": session_id}

    @app.get("/sessions/{session_id}", response_model=SessionInfo)
    async def get_session_info(session_id: str):
        """Get session information."""
        session = await session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        return SessionInfo(
            session_id=session.session_id,
            state=session.state.value,
            buffer_duration_ms=session.buffer_manager.buffer_duration_ms,
            error_message=session.error_message,
        )

    # WebSocket endpoint

    @app.websocket("/ws/tts/{session_id}")
    async def websocket_tts(websocket: WebSocket, session_id: str):
        """
        WebSocket endpoint for streaming TTS.

        Protocol:
        Client sends:
            {"type": "start", "text": "...", "cfg_scale": 3.0, ...}
            {"type": "update", "text": "...", "strategy": "continue"}
            {"type": "stop"}

        Server sends:
            {"type": "audio", "data": "<base64 PCM>", "timestamp_ms": 123, "frame_index": 0}
            {"type": "status", "status": "generating|complete|error", "message": "..."}
        """
        await websocket.accept()

        # Get or create session
        session = await session_manager.get_session(session_id)
        if not session:
            try:
                session = await session_manager.create_session(session_id)
            except (RuntimeError, ValueError) as e:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                })
                await websocket.close()
                return

        # Start audio streaming task
        audio_task = None

        try:
            while True:
                try:
                    data = await asyncio.wait_for(
                        websocket.receive_json(),
                        timeout=0.1,  # Check frequently for audio
                    )
                except asyncio.TimeoutError:
                    # No message, continue streaming audio if active
                    if session.is_active and audio_task is None:
                        audio_task = asyncio.create_task(
                            _stream_audio_to_websocket(websocket, session)
                        )
                    elif audio_task and audio_task.done():
                        # Check for exceptions
                        if audio_task.exception():
                            raise audio_task.exception()
                        audio_task = None
                    continue

                msg_type = data.get("type")

                if msg_type == "start":
                    if session.is_active:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Generation already in progress",
                        })
                        continue

                    await session.start_generation(
                        text=data["text"],
                        cfg_scale=data.get("cfg_scale", config.default_cfg_scale),
                        temperature=data.get("temperature", config.default_temperature),
                        top_p=data.get("top_p", config.default_top_p),
                        cfg_filter_top_k=data.get(
                            "cfg_filter_top_k", config.default_cfg_filter_top_k
                        ),
                    )

                    await websocket.send_json({
                        "type": "status",
                        "status": "generating",
                    })

                    # Start streaming audio
                    audio_task = asyncio.create_task(
                        _stream_audio_to_websocket(websocket, session)
                    )

                elif msg_type == "update":
                    if not session.is_active:
                        await websocket.send_json({
                            "type": "error",
                            "message": "No active generation",
                        })
                        continue

                    await session.update_text(
                        new_text=data["text"],
                        strategy=data.get("strategy", "continue"),
                    )

                elif msg_type == "stop":
                    if audio_task:
                        audio_task.cancel()
                        try:
                            await audio_task
                        except asyncio.CancelledError:
                            pass
                        audio_task = None

                    await session.stop()
                    await websocket.send_json({
                        "type": "status",
                        "status": "stopped",
                    })
                    break

        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {session_id}")
        except Exception as e:
            logger.exception(f"WebSocket error: {e}")
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                })
            except Exception:
                pass
        finally:
            if audio_task:
                audio_task.cancel()
                try:
                    await audio_task
                except asyncio.CancelledError:
                    pass

            await session_manager.close_session(session_id)

    async def _stream_audio_to_websocket(websocket: WebSocket, session):
        """Stream audio chunks to WebSocket client."""
        while session.is_active or not session.buffer_manager.is_complete:
            chunk = await session.get_raw_chunk(timeout=0.5)

            if chunk is None:
                if session.is_complete or session.buffer_manager.is_complete:
                    break
                continue

            # Encode audio as base64
            audio_bytes = chunk.audio_data.astype(np.float32).tobytes()
            audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

            await websocket.send_json({
                "type": "audio",
                "data": audio_b64,
                "timestamp_ms": chunk.timestamp_ms,
                "frame_index": chunk.frame_index,
                "is_final": chunk.is_final,
            })

        # Send completion status
        if session.has_error:
            await websocket.send_json({
                "type": "status",
                "status": "error",
                "message": session.error_message,
            })
        else:
            await websocket.send_json({
                "type": "status",
                "status": "complete",
            })

    # HTTP Streaming endpoint (SSE alternative)

    @app.post("/api/tts/stream")
    async def stream_tts_sse(request: TTSRequest):
        """
        Server-Sent Events endpoint for streaming TTS.

        Returns audio chunks as SSE events.
        """
        session = await session_manager.create_session()

        async def generate_sse():
            try:
                await session.start_generation(
                    text=request.text,
                    cfg_scale=request.cfg_scale,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    cfg_filter_top_k=request.cfg_filter_top_k,
                )

                while session.is_active or not session.buffer_manager.is_complete:
                    chunk = await session.get_raw_chunk(timeout=0.5)

                    if chunk is None:
                        if session.is_complete or session.buffer_manager.is_complete:
                            break
                        continue

                    audio_bytes = chunk.audio_data.astype(np.float32).tobytes()
                    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

                    event_data = json.dumps({
                        "timestamp_ms": chunk.timestamp_ms,
                        "frame_index": chunk.frame_index,
                        "is_final": chunk.is_final,
                        "data": audio_b64,
                    })
                    yield f"event: audio\ndata: {event_data}\n\n"

                # Final event
                if session.has_error:
                    yield f"event: error\ndata: {json.dumps({'message': session.error_message})}\n\n"
                else:
                    yield f"event: complete\ndata: {{}}\n\n"

            finally:
                await session_manager.close_session(session.session_id)

        return StreamingResponse(
            generate_sse(),
            media_type="text/event-stream",
        )

    return app


async def start_tcp_server(
    model,
    config: StreamingConfig | None = None,
    host: str = "0.0.0.0",
    port: int = 8001,
):
    """
    Start a raw TCP server for audio streaming.

    This server accepts connections and streams raw PCM audio bytes.
    Useful for integration with media servers like Asterisk AudioSocket.

    Protocol:
    - Client sends: JSON header with text and params, then newline
    - Server sends: Raw PCM audio bytes (44.1kHz, float32)
    - Client can send additional JSON lines for text updates
    """
    config = config or StreamingConfig()
    manager = TTSSessionManager(model, config)

    async def handle_client(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ):
        """Handle a single TCP client connection."""
        addr = writer.get_extra_info("peername")
        logger.info(f"TCP client connected: {addr}")

        session = None
        try:
            # Read initial request (JSON line)
            line = await reader.readline()
            if not line:
                return

            request = json.loads(line.decode("utf-8"))
            text = request.get("text", "")
            if not text:
                writer.write(b"ERROR: No text provided\n")
                await writer.drain()
                return

            # Create session and start generation
            session = await manager.create_session()
            await session.start_generation(
                text=text,
                cfg_scale=request.get("cfg_scale", config.default_cfg_scale),
                temperature=request.get("temperature", config.default_temperature),
                top_p=request.get("top_p", config.default_top_p),
            )

            # Stream audio
            while session.is_active or not session.buffer_manager.is_complete:
                # Check for text updates (non-blocking read)
                try:
                    reader_task = asyncio.create_task(reader.readline())
                    done, pending = await asyncio.wait(
                        [reader_task],
                        timeout=0.01,
                    )

                    if done:
                        update_line = reader_task.result()
                        if update_line:
                            update = json.loads(update_line.decode("utf-8"))
                            if "text" in update:
                                await session.update_text(
                                    update["text"],
                                    update.get("strategy", "continue"),
                                )
                    else:
                        for task in pending:
                            task.cancel()
                except Exception:
                    pass

                # Get and send audio chunk
                chunk = await session.get_audio_chunk(
                    duration_ms=20.0,
                    timeout=0.1,
                )

                if chunk is not None and len(chunk) > 0:
                    # Send raw PCM bytes
                    audio_bytes = chunk.astype(np.float32).tobytes()
                    writer.write(audio_bytes)
                    await writer.drain()

        except Exception as e:
            logger.exception(f"TCP client error: {e}")
        finally:
            if session:
                await manager.close_session(session.session_id)
            writer.close()
            await writer.wait_closed()
            logger.info(f"TCP client disconnected: {addr}")

    server = await asyncio.start_server(
        handle_client,
        host,
        port,
    )

    addr = server.sockets[0].getsockname()
    logger.info(f"TCP server listening on {addr}")

    async with server:
        await server.serve_forever()


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    tcp_port: int = 8001,
    model=None,
    config: StreamingConfig | None = None,
    model_path: str | None = None,
):
    """
    Run the streaming TTS server.

    Starts both WebSocket (FastAPI) and TCP servers.

    Args:
        host: Host to bind to.
        port: Port for WebSocket server.
        tcp_port: Port for TCP server.
        model: Pre-loaded Dia model (optional).
        config: Streaming configuration.
        model_path: Path to local model checkpoint (.pth file).
    """
    import uvicorn

    config = config or StreamingConfig()

    # Create app
    app = create_app(model, config, model_path=model_path)

    # Run with uvicorn
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parkiet Streaming TTS Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="WebSocket port")
    parser.add_argument("--tcp-port", type=int, default=8001, help="TCP port")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to local model checkpoint (.pth file). "
        "Download from: https://huggingface.co/pevers/parkiet/resolve/main/dia-nl-v1.pth",
    )

    args = parser.parse_args()

    run_server(
        host=args.host,
        port=args.port,
        tcp_port=args.tcp_port,
        model_path=args.model_path,
    )
