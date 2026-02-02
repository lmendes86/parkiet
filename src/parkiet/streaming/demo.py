"""Demo script for streaming TTS generation."""

import asyncio
import time

import numpy as np


async def demo_streaming():
    """Demonstrate streaming TTS generation."""
    print("Loading model...")
    from parkiet.dia.model import Dia
    from parkiet.streaming import StreamingTTSGenerator, StreamingConfig

    model = Dia.from_pretrained("pevers/parkiet", compute_dtype="float32")
    print(f"Model loaded on {model.device}")

    config = StreamingConfig()
    generator = StreamingTTSGenerator(model, config)

    text = "[S1] Hallo, dit is een test van de streaming tekst naar spraak."
    print(f"\nGenerating: {text}")
    print(f"Min buffer latency: {config.min_buffer_latency_ms:.1f}ms")
    print("-" * 60)

    start_time = time.monotonic()
    first_chunk_time = None
    total_samples = 0
    chunk_count = 0

    async for chunk in generator.generate_stream(text):
        if first_chunk_time is None:
            first_chunk_time = time.monotonic()
            latency_ms = (first_chunk_time - start_time) * 1000
            print(f"First chunk latency: {latency_ms:.1f}ms")

        total_samples += len(chunk.audio_data)
        chunk_count += 1

        # Print progress every 10 chunks
        if chunk_count % 10 == 0:
            duration_ms = len(chunk.audio_data) / 44100 * 1000
            print(
                f"Chunk {chunk_count}: {len(chunk.audio_data)} samples "
                f"({duration_ms:.1f}ms), timestamp={chunk.timestamp_ms:.1f}ms"
            )

    total_time = time.monotonic() - start_time
    audio_duration_ms = total_samples / 44100 * 1000

    print("-" * 60)
    print(f"Generation complete!")
    print(f"  Total chunks: {chunk_count}")
    print(f"  Total samples: {total_samples}")
    print(f"  Audio duration: {audio_duration_ms:.1f}ms")
    print(f"  Wall time: {total_time * 1000:.1f}ms")
    print(f"  Real-time factor: {audio_duration_ms / (total_time * 1000):.2f}x")


async def demo_save_audio():
    """Generate and save streaming audio to a file."""
    print("Loading model...")
    from parkiet.dia.model import Dia
    from parkiet.streaming import StreamingTTSGenerator, StreamingConfig
    import soundfile as sf

    model = Dia.from_pretrained("pevers/parkiet", compute_dtype="float32")
    print(f"Model loaded on {model.device}")

    config = StreamingConfig()
    generator = StreamingTTSGenerator(model, config)

    text = "[S1] Dit is een test van de streaming tekst naar spraak functionaliteit."
    print(f"\nGenerating: {text}")

    # Collect all audio chunks
    audio_chunks = []
    async for chunk in generator.generate_stream(text):
        audio_chunks.append(chunk.audio_data)
        print(".", end="", flush=True)

    print("\n")

    # Concatenate and save
    if audio_chunks:
        audio = np.concatenate(audio_chunks)
        output_path = "streaming_output.wav"
        sf.write(output_path, audio, 44100)
        print(f"Saved audio to {output_path}")
        print(f"Duration: {len(audio) / 44100:.2f}s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Streaming TTS Demo")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save output to file instead of just timing",
    )
    args = parser.parse_args()

    if args.save:
        asyncio.run(demo_save_audio())
    else:
        asyncio.run(demo_streaming())
