"""Text update handler for mid-generation text changes."""

import torch

from parkiet.dia.state import EncoderInferenceState

from .generation_state import StreamingGenerationState, TextUpdate


class TextUpdateHandler:
    """
    Handles text re-encoding and state updates during streaming generation.

    When new text arrives during generation, this handler:
    1. Re-encodes the new text
    2. Re-runs the encoder
    3. Recomputes the cross-attention K/V cache
    4. Updates the generation state based on the chosen strategy

    Strategies:
    - 'continue': Keep decoder self-attention cache, only update cross-attention.
                  This preserves prosody and rhythm continuity.
    - 'restart': Discard all decoder state and start fresh. Higher quality
                 but introduces a gap in the audio stream.
    """

    def __init__(self, model):
        """
        Initialize the text update handler.

        Args:
            model: The Dia TTS model.
        """
        self.model = model

    async def apply_update(
        self,
        state: StreamingGenerationState,
        update: TextUpdate,
    ) -> StreamingGenerationState:
        """
        Apply a text update to the generation state.

        Args:
            state: Current generation state.
            update: Text update to apply.

        Returns:
            Updated generation state.
        """
        if update.strategy == "restart":
            return await self._apply_restart(state, update)
        else:  # "continue" strategy (default)
            return await self._apply_continue(state, update)

    async def _apply_continue(
        self,
        state: StreamingGenerationState,
        update: TextUpdate,
    ) -> StreamingGenerationState:
        """
        Apply text update using 'continue' strategy.

        This strategy:
        - Re-encodes the new text
        - Recomputes encoder output and cross-attention K/V cache
        - Keeps decoder self-attention cache (preserves prosody)
        - Updates cross-attention references in decoder state
        """
        device = state.device
        batch_size = 1

        # Re-encode new text
        new_text_tokens = self.model._encode_text(update.new_text)
        padded_text = self.model._pad_text_input([new_text_tokens])

        # Create encoder input (unconditional + conditional for CFG)
        enc_input_uncond = torch.zeros_like(padded_text)
        enc_input_cond = padded_text
        stacked_inputs = torch.stack([enc_input_uncond, enc_input_cond], dim=1)
        enc_input = stacked_inputs.view(2 * batch_size, -1)

        # Re-run encoder
        new_enc_state = EncoderInferenceState.new(self.model.config, enc_input_cond)
        new_encoder_out = self.model.model.encoder(enc_input, new_enc_state)

        # Recompute cross-attention K/V cache (this is mandatory)
        new_cross_attn_cache = self.model.model.decoder.precompute_cross_attn_cache(
            new_encoder_out
        )

        # Update state - KEEP decoder self-attention cache
        state.text = update.new_text
        state.text_tokens = new_text_tokens
        state.text_version += 1

        state.enc_state = new_enc_state
        state.encoder_out = new_encoder_out
        state.cross_attn_cache = new_cross_attn_cache

        # Update decoder state's cross-attention cache references
        state.dec_state.cross_attn_cache = new_cross_attn_cache
        state.dec_state.enc_out = new_encoder_out

        # Update cross-attention mask for new text length
        dec_mask = torch.ones((2 * batch_size, 1), dtype=torch.bool, device=device)
        from parkiet.dia.state import create_attn_mask

        new_cross_attn_mask = create_attn_mask(
            dec_mask, new_enc_state.padding_mask, device, is_causal=False
        )
        state.dec_state.cross_attn_mask = new_cross_attn_mask

        return state

    async def _apply_restart(
        self,
        state: StreamingGenerationState,
        update: TextUpdate,
    ) -> StreamingGenerationState:
        """
        Apply text update using 'restart' strategy.

        This strategy completely reinitializes the generation with the new text.
        Higher quality but creates a discontinuity in the audio stream.
        """
        # Import here to avoid circular dependency
        from .generator import StreamingTTSGenerator

        # Create a new generator and initialize fresh state
        # Note: This is a simplified implementation. In practice,
        # you might want to preserve some audio context.

        device = state.device
        batch_size = 1

        # Re-encode new text
        new_text_tokens = self.model._encode_text(update.new_text)
        padded_text = self.model._pad_text_input([new_text_tokens])

        # Create encoder input
        enc_input_uncond = torch.zeros_like(padded_text)
        enc_input_cond = padded_text
        stacked_inputs = torch.stack([enc_input_uncond, enc_input_cond], dim=1)
        enc_input = stacked_inputs.view(2 * batch_size, -1)

        # Run encoder
        new_enc_state = EncoderInferenceState.new(self.model.config, enc_input_cond)
        new_encoder_out = self.model.model.encoder(enc_input, new_enc_state)

        # Precompute cross-attention cache
        new_cross_attn_cache = self.model.model.decoder.precompute_cross_attn_cache(
            new_encoder_out
        )

        # Create new decoder state
        from parkiet.dia.state import DecoderInferenceState, DecoderOutput

        new_dec_state = DecoderInferenceState.new(
            self.model.config,
            new_enc_state,
            new_encoder_out,
            new_cross_attn_cache,
            self.model.compute_dtype,
            max_generation_length=self.model.config.decoder_config.max_position_embeddings,
        )

        # Initialize decoder output
        new_dec_output = DecoderOutput.new(batch_size, self.model.config, device)

        # Prefill with BOS
        from parkiet.dia.audio import build_delay_indices, apply_audio_delay

        delay_pattern = self.model.config.delay_pattern
        max_delay_pattern = max(delay_pattern)
        num_channels = self.model.config.decoder_config.num_channels
        audio_bos_value = self.model.config.bos_token_id

        prefill = torch.full(
            (batch_size, 1, num_channels),
            fill_value=audio_bos_value,
            dtype=torch.int,
            device=device,
        )

        delay_precomp = build_delay_indices(
            B=batch_size,
            T=1,
            C=num_channels,
            delay_pattern=delay_pattern,
        )

        delayed_batch = apply_audio_delay(
            audio_BxTxC=prefill,
            pad_value=-1,
            bos_value=audio_bos_value,
            precomp=delay_precomp,
        )

        new_dec_output.prefill(delayed_batch, [1])

        # Update state with fresh decoder
        state.text = update.new_text
        state.text_tokens = new_text_tokens
        state.text_version += 1

        state.enc_state = new_enc_state
        state.encoder_out = new_encoder_out
        state.cross_attn_cache = new_cross_attn_cache

        state.dec_state = new_dec_state
        state.dec_output = new_dec_output
        state.current_step = 0
        state.prefill_steps = [1]

        # Reset EOS state
        state.eos_detected = False
        state.eos_countdown = -1
        state.finished_step = -1

        return state
