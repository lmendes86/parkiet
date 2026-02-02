"""WebSocket client for testing the streaming TTS server."""

import argparse
import asyncio
import base64
import json
import struct
import sys
import time
import queue

import numpy as np


def play_audio_realtime(audio_queue: queue.Queue, sample_rate: int = 44100):
    """
    Play audio chunks in real-time as they arrive.

    Args:
        audio_queue: Queue receiving numpy float32 audio chunks
        sample_rate: Audio sample rate
    """
    try:
        import sounddevice as sd
    except ImportError:
        print("Note: Install sounddevice for real-time playback: uv pip install sounddevice")
        return

    stream = sd.OutputStream(
        samplerate=sample_rate,
        channels=1,
        dtype=np.float32,
        blocksize=1024,
    )
    stream.start()

    try:
        while True:
            try:
                chunk = audio_queue.get(timeout=2.0)
                if chunk is None:  # Sentinel to stop
                    break
                stream.write(chunk)
            except queue.Empty:
                continue
    finally:
        stream.stop()
        stream.close()


async def test_streaming_tts(
    url: str,
    text: str,
    output_file: str | None = None,
    update_text: str | None = None,
    update_delay: float = 2.0,
    play_audio: bool = False,
):
    """
    Test the WebSocket streaming TTS endpoint.

    Args:
        url: WebSocket URL (e.g., ws://localhost:8000/ws/tts/test-session)
        text: Initial text to synthesize
        output_file: Optional path to save audio as WAV
        update_text: Optional text to send as update mid-generation
        update_delay: Delay in seconds before sending update
        play_audio: If True, play audio in real-time as it arrives
    """
    try:
        import websockets
    except ImportError:
        print("Please install websockets: uv pip install websockets")
        sys.exit(1)

    # Setup audio playback if requested
    audio_queue: queue.Queue | None = None
    playback_thread = None
    if play_audio:
        audio_queue = queue.Queue()
        import threading
        playback_thread = threading.Thread(
            target=play_audio_realtime,
            args=(audio_queue,),
            daemon=True,
        )
        playback_thread.start()

    audio_chunks = []
    start_time = time.monotonic()
    first_audio_time = None
    total_frames = 0

    print(f"Connecting to {url}...")

    try:
        async with websockets.connect(url) as ws:
            print("Connected!")

            # Send start message
            start_msg = {
                "type": "start",
                "text": text,
                "cfg_scale": 3.0,
                "temperature": 1.2,
                "top_p": 0.95,
            }
            print(f"Sending: {json.dumps(start_msg)}")
            await ws.send(json.dumps(start_msg))

            # Schedule text update if requested
            update_task = None
            if update_text:
                update_task = asyncio.create_task(
                    send_text_update(ws, update_text, update_delay)
                )

            # Receive and process messages
            first_audio_time, total_frames = await receive_audio_stream(
                ws, audio_chunks, audio_queue, start_time
            )

            if update_task:
                update_task.cancel()
                try:
                    await update_task
                except asyncio.CancelledError:
                    pass

    finally:
        # Signal playback thread to stop
        if audio_queue:
            audio_queue.put(None)
        if playback_thread:
            playback_thread.join(timeout=3.0)

    # Print summary and save file
    print_summary(audio_chunks, start_time, first_audio_time, total_frames, output_file)


async def send_text_update(ws, update_text: str, delay: float):
    """Send a text update after a delay."""
    await asyncio.sleep(delay)
    update_msg = {
        "type": "update",
        "text": update_text,
        "strategy": "continue",
    }
    print(f"\nSending update: {json.dumps(update_msg)}")
    await ws.send(json.dumps(update_msg))


async def receive_audio_stream(ws, audio_chunks: list, audio_queue, start_time: float):
    """Receive audio chunks from WebSocket and process them."""
    import websockets

    first_audio_time = None
    total_frames = 0

    try:
        while True:
            msg = await ws.recv()
            data = json.loads(msg)

            if data["type"] == "audio":
                first_audio_time, total_frames = process_audio_chunk(
                    data, audio_chunks, audio_queue,
                    first_audio_time, total_frames, start_time
                )

            elif data["type"] == "status":
                print(f"\nStatus: {data['status']}")
                if data["status"] in ("complete", "error", "stopped"):
                    if "message" in data and data["message"]:
                        print(f"Message: {data['message']}")
                    break

            elif data["type"] == "error":
                print(f"\nError: {data['message']}")
                break

    except websockets.ConnectionClosed as e:
        print(f"\nConnection closed: {e}")

    return first_audio_time, total_frames


def process_audio_chunk(data, audio_chunks, audio_queue, first_audio_time, total_frames, start_time):
    """Process a single audio chunk."""
    if first_audio_time is None:
        first_audio_time = time.monotonic()
        latency_ms = (first_audio_time - start_time) * 1000
        print(f"First audio chunk latency: {latency_ms:.1f}ms")

    # Decode base64 audio
    audio_bytes = base64.b64decode(data["data"])
    audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
    audio_chunks.append(audio_data)

    # Send to playback queue if active
    if audio_queue:
        audio_queue.put(audio_data.copy())

    total_frames += 1
    duration_ms = len(audio_data) / 44100 * 1000

    # Diagnostic info for first few chunks
    if total_frames <= 3:
        print(
            f"\n  Chunk {total_frames}: {len(audio_data)} samples ({duration_ms:.1f}ms), "
            f"range=[{audio_data.min():.4f}, {audio_data.max():.4f}], "
            f"frame_index={data['frame_index']}"
        )
    elif total_frames % 10 == 0:
        print(
            f"  Frame {data['frame_index']}: "
            f"{len(audio_data)} samples ({duration_ms:.1f}ms), "
            f"timestamp={data['timestamp_ms']:.1f}ms"
        )
    else:
        print(".", end="", flush=True)

    return first_audio_time, total_frames


def print_summary(audio_chunks, start_time, first_audio_time, total_frames, output_file):
    """Print summary and optionally save audio to file."""
    total_time = time.monotonic() - start_time
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Total frames received: {total_frames}")

    if audio_chunks:
        all_audio = np.concatenate(audio_chunks)
        audio_duration = len(all_audio) / 44100
        print(f"  Total samples: {len(all_audio)}")
        print(f"  Audio duration: {audio_duration:.2f}s")
        print(f"  Wall time: {total_time:.2f}s")
        print(f"  Real-time factor: {audio_duration / total_time:.2f}x")

        if first_audio_time:
            print(f"  First chunk latency: {(first_audio_time - start_time) * 1000:.1f}ms")

        # Save to file if requested
        if output_file:
            try:
                import soundfile as sf
                sf.write(output_file, all_audio, 44100)
                print(f"  Saved to: {output_file}")
            except ImportError:
                save_wav(output_file, all_audio, 44100)
                print(f"  Saved to: {output_file}")


def save_wav(path: str, audio: np.ndarray, sample_rate: int):
    """Save audio as WAV file without soundfile dependency."""
    audio_int16 = (audio * 32767).astype(np.int16)
    with open(path, "wb") as f:
        # WAV header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(audio_int16) * 2))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))  # Subchunk1Size
        f.write(struct.pack("<H", 1))  # AudioFormat (PCM)
        f.write(struct.pack("<H", 1))  # NumChannels
        f.write(struct.pack("<I", sample_rate))  # SampleRate
        f.write(struct.pack("<I", sample_rate * 2))  # ByteRate
        f.write(struct.pack("<H", 2))  # BlockAlign
        f.write(struct.pack("<H", 16))  # BitsPerSample
        f.write(b"data")
        f.write(struct.pack("<I", len(audio_int16) * 2))
        f.write(audio_int16.tobytes())


async def main():
    parser = argparse.ArgumentParser(description="Test WebSocket streaming TTS")
    parser.add_argument(
        "--host",
        default="localhost",
        help="Server host (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )
    parser.add_argument(
        "--session-id",
        default="test-session",
        help="Session ID (default: test-session)",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Full WebSocket URL (overrides --host, --port, --session-id)",
    )
    parser.add_argument(
        "--text",
        default="[S1] Hallo, dit is een test van de streaming tekst naar spraak.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output WAV file path",
    )
    parser.add_argument(
        "--update-text",
        type=str,
        default=None,
        help="Text to send as update mid-generation",
    )
    parser.add_argument(
        "--update-delay",
        type=float,
        default=2.0,
        help="Delay before sending text update (seconds)",
    )
    parser.add_argument(
        "--play",
        "-p",
        action="store_true",
        help="Play audio in real-time (requires sounddevice)",
    )

    args = parser.parse_args()

    # Build URL from components or use explicit URL
    if args.url:
        url = args.url
    else:
        url = f"ws://{args.host}:{args.port}/ws/tts/{args.session_id}"

    await test_streaming_tts(
        url=url,
        text=args.text,
        output_file=args.output,
        update_text=args.update_text,
        update_delay=args.update_delay,
        play_audio=args.play,
    )


if __name__ == "__main__":
    asyncio.run(main())
