import math

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Transformer2DModel

from .kernels.residual import gated_add


class ImageTransformer(QwenImage21Transformer2DModel):
    def forward(self, hidden_states, encoder_hidden_states, timestep, img_shapes, img_mask,
                encoder_hidden_states_mask=None, attention_kwargs=None, kv_cache=None,
                kv_cache_mode=None, return_dict=True):
        tokens = math.prod(img_shapes[0][-1])
        if kv_cache_mode == "cached":
            output = self._decode_target(hidden_states, timestep, tokens, kv_cache)
            return Transformer2DModelOutput(sample=output) if return_dict else (output,)

        def remember_layout(module, args, kwargs):
            valid_keys = kwargs["key_valid"]
            kv_cache.target_rotation = kwargs["rotary_emb"][-tokens:]
            kv_cache.attention_mask = None if valid_keys is None else valid_keys[:, None, None, :]
            kv_cache.target_mask = kwargs["target_token_mask"][-tokens:]

        hook = None
        if kv_cache_mode == "extract":
            hook = self.transformer_blocks[0].register_forward_pre_hook(remember_layout, with_kwargs=True)
        try:
            return super().forward(
                hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states,
                timestep=timestep, img_shapes=img_shapes, img_mask=img_mask,
                encoder_hidden_states_mask=encoder_hidden_states_mask, attention_kwargs=attention_kwargs,
                kv_cache=kv_cache, kv_cache_mode=kv_cache_mode, return_dict=return_dict,
            )
        finally:
            if hook is not None:
                hook.remove()

    def _decode_target(self, hidden_states, timestep, tokens, cache):
        hidden = self.img_in(hidden_states)[:, -tokens:].contiguous()
        timestep = timestep.to(hidden.dtype)
        temb = self.time_text_embed(torch.cat((timestep, timestep.new_zeros(1))), hidden)
        modulation = self.modulation(temb)[:-1].unsqueeze(1)
        scale1, gate1, scale2, gate2 = modulation.chunk(4, dim=-1)
        scale1, scale2 = 1 + scale1, 1 + scale2
        gate1, gate2 = gate1.tanh().contiguous(), gate2.tanh().contiguous()
        for index, block in enumerate(self.transformer_blocks):
            attention = block.attn(
                hidden_states=block.img_norm1(hidden) * scale1, attention_mask=cache.attention_mask,
                rotary_emb=cache.target_rotation, layer_cache=cache.get_layer(index), kv_cache_mode="cached",
            )
            hidden = gated_add(hidden, attention, gate1)
            mlp = block.img_mlp(block.img_norm2(hidden) * scale2)
            hidden = gated_add(hidden, mlp, gate2)
        return self.proj_out(self.norm_out(hidden, temb, cache.target_mask))
