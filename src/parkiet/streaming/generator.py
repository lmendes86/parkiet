"""Streaming TTS generator for real-time audio output."""

import asyncio
import time
import uuid
from collections.abc import AsyncGenerator

import numpy as np
import torch
import torch.nn.functional as F

from parkiet.dia.audio import build_delay_indices, apply_audio_delay
from parkiet.dia.state import (
    EncoderInferenceState,
    DecoderInferenceState,
    DecoderOutput,
)

from .config import StreamingConfig
from .dac_decoder import IncrementalDACDecoder
from .generation_state import (
    StreamingGenerationState,
    StreamingChunk,
    TextUpdate,
    GenerationStatus,
)


def _sample_next_token(
    logits_BCxV: torch.Tensor,
    temperature: float,
    top_p: float,
    top_k: int | None,
    audio_eos_value: int,
) -> torch.Tensor:
    """Sample next token from logits with temperature, top-p, and top-k."""
    if temperature == 0.0:
        return torch.argmax(logits_BCxV, dim=-1)

    logits_BCxV = logits_BCxV / temperature

    # EOS handling
    if audio_eos_value is not None and audio_eos_value >= 0:
        top_logit_indices_BC = torch.argmax(logits_BCxV, dim=-1)
        eos_not_highest_mask_BC = top_logit_indices_BC != audio_eos_value
        mask_eos_unless_highest_BCxV = torch.zeros_like(logits_BCxV, dtype=torch.bool)
        mask_eos_unless_highest_BCxV[eos_not_highest_mask_BC, audio_eos_value] = True
        logits_BCxV = logits_BCxV.masked_fill(mask_eos_unless_highest_BCxV, -torch.inf)
        eos_highest_mask_BC = top_logit_indices_BC == audio_eos_value
        mask_eos_highest_BCxV = torch.zeros_like(logits_BCxV, dtype=torch.bool)
        mask_eos_highest_BCxV[eos_highest_mask_BC, :audio_eos_value] = True
        logits_BCxV = logits_BCxV.masked_fill(mask_eos_highest_BCxV, -torch.inf)

    # Top-k filtering
    if top_k is not None:
        _, top_k_indices_BCxV = torch.topk(logits_BCxV, k=top_k, dim=-1)
        mask = torch.ones_like(logits_BCxV, dtype=torch.bool)
        mask = mask.scatter(dim=-1, index=top_k_indices_BCxV, value=False)
        logits_BCxV = logits_BCxV.masked_fill(mask, -torch.inf)

    # Top-p (nucleus) sampling
    if top_p < 1.0:
        probs_BCxV = torch.softmax(logits_BCxV, dim=-1)
        sorted_probs_BCxV, sorted_indices_BCxV = torch.sort(
            probs_BCxV, dim=-1, descending=True
        )
        cumulative_probs_BCxV = torch.cumsum(sorted_probs_BCxV, dim=-1)

        sorted_indices_to_remove_BCxV = cumulative_probs_BCxV > top_p
        sorted_indices_to_remove_BCxV = torch.roll(
            sorted_indices_to_remove_BCxV, shifts=1, dims=-1
        )
        sorted_indices_to_remove_BCxV[..., 0] = torch.zeros_like(
            sorted_indices_to_remove_BCxV[..., 0]
        )

        indices_to_remove_BCxV = torch.zeros_like(sorted_indices_to_remove_BCxV)
        indices_to_remove_BCxV = indices_to_remove_BCxV.scatter(
            dim=-1, index=sorted_indices_BCxV, src=sorted_indices_to_remove_BCxV
        )
        logits_BCxV = logits_BCxV.masked_fill(indices_to_remove_BCxV, -torch.inf)

    final_probs_BCxV = torch.softmax(logits_BCxV, dim=-1)
    sampled_indices_BC = torch.multinomial(final_probs_BCxV, num_samples=1)
    return sampled_indices_BC.squeeze(-1)


class StreamingTTSGenerator:
    """
    Async generator for streaming TTS output.

    This class converts the batch TTS generation into an async generator
    that yields audio chunks incrementally as tokens are generated.

    Example usage:
        generator = StreamingTTSGenerator(model)
        async for chunk in generator.generate_stream("[S1] Hello world"):
            # Process audio chunk
            play_audio(chunk.audio_data)
    """

    def __init__(
        self,
        model,  # Dia model
        config: StreamingConfig | None = None,
    ):
        """
        Initialize the streaming generator.

        Args:
            model: The Dia TTS model.
            config: Streaming configuration. Uses defaults if not provided.
        """
        self.model = model
        self.config = config or StreamingConfig()

        # Update config from model's config
        if hasattr(model, "config"):
            self.config.delay_pattern = model.config.delay_pattern
            self.config.num_channels = model.config.decoder_config.num_channels

        self.config.validate()

        # Create incremental DAC decoder
        self.dac_decoder = IncrementalDACDecoder(
            model.dac_model,
            self.config,
            model.device,
        )

    async def generate_stream(
        self,
        text: str,
        cfg_scale: float | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        cfg_filter_top_k: int | None = None,
        max_tokens: int | None = None,
        audio_prompt: torch.Tensor | None = None,
    ) -> AsyncGenerator[StreamingChunk, TextUpdate | None]:
        """
        Generate audio stream from text.

        This is an async generator that yields StreamingChunk objects
        containing audio data. It can also receive TextUpdate objects
        via .asend() for mid-generation text updates.

        Args:
            text: Input text to synthesize.
            cfg_scale: Classifier-free guidance scale.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            cfg_filter_top_k: Top-k for CFG filtering.
            max_tokens: Maximum generation length.
            audio_prompt: Optional audio prompt tensor.

        Yields:
            StreamingChunk: Audio chunks as they're decoded.

        Receives:
            TextUpdate: Text updates to apply mid-generation.
        """
        # Use defaults from config
        cfg_scale = cfg_scale or self.config.default_cfg_scale
        temperature = temperature or self.config.default_temperature
        top_p = top_p or self.config.default_top_p
        cfg_filter_top_k = cfg_filter_top_k or self.config.default_cfg_filter_top_k
        max_tokens = max_tokens or self.config.max_tokens

        # Initialize state
        state = await self._initialize_generation(
            text=text,
            audio_prompt=audio_prompt,
            max_tokens=max_tokens,
        )

        # Reset DAC decoder
        self.dac_decoder.reset()

        start_time = time.monotonic()
        frame_index = 0

        try:
            while state.is_active:
                # Allow asyncio to yield control
                await asyncio.sleep(0)

                # Generate one step
                pred_BxC = await self._generate_step(
                    state=state,
                    cfg_scale=cfg_scale,
                    temperature=temperature,
                    top_p=top_p,
                    cfg_filter_top_k=cfg_filter_top_k,
                )

                if pred_BxC is None:
                    break

                # Handle EOS
                self._handle_eos(state, pred_BxC, max_tokens)

                # Update decoder output
                state.dec_output.update_one(
                    pred_BxC,
                    state.current_step + 1,
                    apply_mask=not self._is_bos_over(state),
                )

                state.current_step += 1

                # Add frame to DAC decoder
                # pred_BxC is [B, C], we only support B=1 for streaming
                audio = self.dac_decoder.add_frame(pred_BxC[0])

                if audio is not None:
                    timestamp_ms = (time.monotonic() - start_time) * 1000
                    state.update_audio_stats(len(audio))

                    chunk = StreamingChunk(
                        audio_data=audio,
                        timestamp_ms=timestamp_ms,
                        frame_index=frame_index,
                        is_final=False,
                    )
                    frame_index += 1

                    # Yield chunk and potentially receive text update
                    text_update = yield chunk

                    if text_update is not None:
                        state = await self._handle_text_update(state, text_update)

            # Flush remaining audio
            final_audio = self.dac_decoder.flush()
            if final_audio is not None and len(final_audio) > 0:
                timestamp_ms = (time.monotonic() - start_time) * 1000
                state.update_audio_stats(len(final_audio))

                yield StreamingChunk(
                    audio_data=final_audio,
                    timestamp_ms=timestamp_ms,
                    frame_index=frame_index,
                    is_final=True,
                )

        except Exception as e:
            state.set_error(str(e))
            raise

    async def _initialize_generation(
        self,
        text: str,
        audio_prompt: torch.Tensor | None,
        max_tokens: int,
    ) -> StreamingGenerationState:
        """Initialize generation state."""
        device = self.model.device
        batch_size = 1  # Streaming only supports batch size 1

        # Encode text
        text_tokens = self.model._encode_text(text)
        padded_text = self.model._pad_text_input([text_tokens])

        # Create encoder input (unconditional + conditional for CFG)
        enc_input_uncond = torch.zeros_like(padded_text)
        enc_input_cond = padded_text
        stacked_inputs = torch.stack([enc_input_uncond, enc_input_cond], dim=1)
        enc_input = stacked_inputs.view(2 * batch_size, -1)

        # Run encoder
        enc_state = EncoderInferenceState.new(self.model.config, enc_input_cond)
        encoder_out = self.model.model.encoder(enc_input, enc_state)

        # Precompute cross-attention cache
        cross_attn_cache = self.model.model.decoder.precompute_cross_attn_cache(
            encoder_out
        )

        # Initialize decoder state
        dec_state = DecoderInferenceState.new(
            self.model.config,
            enc_state,
            encoder_out,
            cross_attn_cache,
            self.model.compute_dtype,
            max_generation_length=max_tokens,
        )

        # Prepare audio prompt
        audio_prompts = [audio_prompt] if audio_prompt is not None else [None]
        prefill, prefill_steps = self._prepare_audio_prompt(audio_prompts)

        # Initialize decoder output
        dec_output = DecoderOutput.new(batch_size, self.model.config, device)
        dec_output.prefill(prefill, prefill_steps)

        # Prefill decoder if we have audio prompt
        dec_step = min(prefill_steps) - 1
        if dec_step > 0:
            dec_state.prepare_step(0, dec_step)
            tokens_BxTxC = dec_output.get_tokens_at(0, dec_step).repeat_interleave(
                2, dim=0
            )
            self.model.model.decoder.forward(tokens_BxTxC, dec_state)

        # Create state object
        state = StreamingGenerationState(
            generation_id=str(uuid.uuid4()),
            text=text,
            text_tokens=text_tokens,
            enc_state=enc_state,
            encoder_out=encoder_out,
            cross_attn_cache=cross_attn_cache,
            dec_state=dec_state,
            dec_output=dec_output,
            current_step=dec_step,
            prefill_steps=prefill_steps,
            status=GenerationStatus.GENERATING,
            device=device,
        )

        return state

    def _prepare_audio_prompt(
        self, audio_prompts: list[torch.Tensor | None]
    ) -> tuple[torch.Tensor, list[int]]:
        """Prepare audio prompt tensor for decoder."""
        num_channels = self.model.config.decoder_config.num_channels
        audio_bos_value = self.model.config.bos_token_id
        delay_pattern = self.model.config.delay_pattern
        max_delay_pattern = max(delay_pattern)
        batch_size = len(audio_prompts)
        device = self.model.device

        max_len = (
            max(p.shape[0] if p is not None else 0 for p in audio_prompts)
            + max_delay_pattern
        )
        prefill_steps = []

        prefill = torch.full(
            (batch_size, max_len, num_channels),
            fill_value=-1,
            dtype=torch.int,
            device=device,
        )

        prefill[:, 0, :] = audio_bos_value

        for i in range(batch_size):
            prompt = audio_prompts[i]
            if prompt is not None:
                prompt = prompt.to(device=device, dtype=torch.int)
                prefill[i, 1 : prompt.shape[0] + 1, :] = prompt
                prefill_steps.append(prompt.shape[0] + 1)
            else:
                prefill_steps.append(1)

        delay_precomp = build_delay_indices(
            B=batch_size,
            T=max_len,
            C=num_channels,
            delay_pattern=delay_pattern,
        )

        delayed_batch = apply_audio_delay(
            audio_BxTxC=prefill,
            pad_value=-1,
            bos_value=audio_bos_value,
            precomp=delay_precomp,
        )

        return delayed_batch, prefill_steps

    async def _generate_step(
        self,
        state: StreamingGenerationState,
        cfg_scale: float,
        temperature: float,
        top_p: float,
        cfg_filter_top_k: int,
    ) -> torch.Tensor | None:
        """Generate one token step."""
        dec_state = state.dec_state
        dec_output = state.dec_output
        current_step = state.current_step

        # Check completion
        if state.eos_countdown == 0:
            state.status = GenerationStatus.COMPLETE
            return None

        # Prepare decoder step
        dec_state.prepare_step(current_step)

        # Get tokens for this step
        tokens_Bx1xC = dec_output.get_tokens_at(current_step).repeat_interleave(
            2, dim=0
        )

        # Run decoder
        audio_eos_value = self.model.config.eos_token_id
        logits_Bx1xCxV = self.model.model.decoder.decode_step(
            tokens_Bx1xC,
            dec_state,
            torch.tensor([current_step], device=state.device),
        )

        # Apply CFG and sample
        pred_BxC = self._apply_cfg_and_sample(
            logits_Bx1xCxV,
            cfg_scale=cfg_scale,
            temperature=temperature,
            top_p=top_p,
            cfg_filter_top_k=cfg_filter_top_k,
            audio_eos_value=audio_eos_value,
        )

        return pred_BxC

    def _apply_cfg_and_sample(
        self,
        logits_Bx1xCxV: torch.Tensor,
        cfg_scale: float,
        temperature: float,
        top_p: float,
        cfg_filter_top_k: int,
        audio_eos_value: int,
    ) -> torch.Tensor:
        """Apply classifier-free guidance and sample tokens."""
        B = logits_Bx1xCxV.shape[0] // 2
        num_channels = self.model.config.decoder_config.num_channels

        logits_last_2BxCxV = logits_Bx1xCxV[:, -1]
        logits_last_Bx2xCxV = logits_last_2BxCxV.view(B, 2, *logits_last_2BxCxV.shape[1:])

        uncond_logits_BxCxV = logits_last_Bx2xCxV[:, 0, :, :]
        cond_logits_BxCxV = logits_last_Bx2xCxV[:, 1, :, :]

        # CFG-filter: Use CFG logits to determine top-k mask
        cfg_logits_BxCxV = cond_logits_BxCxV + cfg_scale * (
            cond_logits_BxCxV - uncond_logits_BxCxV
        )

        _, top_k_indices_BxCxk = torch.topk(cfg_logits_BxCxV, k=cfg_filter_top_k, dim=-1)
        mask_BxCxV = torch.ones_like(cond_logits_BxCxV, dtype=torch.bool)
        mask_BxCxV = mask_BxCxV.scatter(dim=-1, index=top_k_indices_BxCxk, value=False)
        logits_BxCxV = cond_logits_BxCxV.masked_fill(mask_BxCxV, -torch.inf)

        # Constrain vocabulary
        logits_BxCxV[:, :, audio_eos_value + 1 :] = torch.full_like(
            logits_BxCxV[:, :, audio_eos_value + 1 :],
            fill_value=-torch.inf,
        )
        logits_BxCxV[:, 1:, audio_eos_value:] = torch.full_like(
            logits_BxCxV[:, 1:, audio_eos_value:],
            fill_value=-torch.inf,
        )

        # Sample
        flat_logits_BCxV = logits_BxCxV.view(B * num_channels, -1)
        pred_BC = _sample_next_token(
            flat_logits_BCxV.float(),
            temperature=temperature,
            top_p=top_p,
            top_k=cfg_filter_top_k,
            audio_eos_value=audio_eos_value,
        )

        return pred_BC.view(B, num_channels)

    def _handle_eos(
        self,
        state: StreamingGenerationState,
        pred_BxC: torch.Tensor,
        max_tokens: int,
    ) -> None:
        """Handle EOS detection and countdown."""
        audio_eos_value = self.model.config.eos_token_id
        audio_pad_value = self.model.config.pad_token_id
        max_delay = self.config.max_delay
        delay_pattern = torch.tensor(
            self.config.delay_pattern, device=state.device, dtype=torch.long
        )
        current_step = state.current_step + 1

        # Check for EOS in channel 0
        is_eos_token = (not state.eos_detected) and (
            pred_BxC[0, 0].item() == audio_eos_value
        )
        is_max_len = current_step >= max_tokens - max_delay

        if is_eos_token or is_max_len:
            state.mark_eos_detected(current_step, max_delay)

        # Handle EOS countdown padding
        if state.eos_countdown > 0:
            step_after_eos = max_delay - state.eos_countdown

            for c in range(self.config.num_channels):
                delay = self.config.delay_pattern[c]
                if step_after_eos == delay:
                    pred_BxC[0, c] = audio_eos_value
                elif step_after_eos > delay:
                    pred_BxC[0, c] = audio_pad_value

            state.decrement_eos_countdown()

    def _is_bos_over(self, state: StreamingGenerationState) -> bool:
        """Check if BOS phase is complete."""
        max_delay = self.config.max_delay
        for prefill_step in state.prefill_steps:
            if state.current_step - prefill_step <= max_delay:
                return False
        return True

    async def _handle_text_update(
        self,
        state: StreamingGenerationState,
        update: TextUpdate,
    ) -> StreamingGenerationState:
        """
        Handle a text update during generation.

        This re-encodes the text and updates the cross-attention cache
        while optionally preserving the decoder self-attention cache.
        """
        from .text_updater import TextUpdateHandler

        handler = TextUpdateHandler(self.model)
        return await handler.apply_update(state, update)
