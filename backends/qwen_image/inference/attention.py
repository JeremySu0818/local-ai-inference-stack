import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21AttnProcessor

from .kernels.linear import VERIFIED_ROWS, project
from .kernels.projections import project_qkv
from .kernels.rotary import normalize_rotate


class CachedAttention(QwenImage21AttnProcessor):
    def __call__(self, attn, hidden_states, attention_mask=None, rotary_emb=None,
                 layer_cache=None, kv_cache_mode=None, **kwargs):
        if kv_cache_mode != "cached":
            return super().__call__(
                attn, hidden_states, attention_mask=attention_mask, rotary_emb=rotary_emb,
                layer_cache=layer_cache, kv_cache_mode=kv_cache_mode, **kwargs,
            )
        shape = (attn.heads, attn.inner_dim // attn.heads)
        prefix_key, prefix_value = layer_cache.get()
        batch, tokens, channels = hidden_states.shape
        if batch == 1 and tokens in VERIFIED_ROWS and channels == attn.inner_dim == 4096 and hidden_states.is_contiguous():
            prefix = prefix_key.shape[1]
            key = hidden_states.new_empty((1, prefix + tokens, *shape))
            value = torch.empty_like(key)
            key[:, :prefix].copy_(prefix_key)
            value[:, :prefix].copy_(prefix_value)
            raw_query, raw_key = project_qkv(attn, hidden_states, value[:, prefix:])
            normalize_rotate(raw_key.unflatten(-1, shape), attn.norm_k, rotary_emb, destination=key[:, prefix:])
        else:
            raw_query = project(attn.to_q, hidden_states)
            raw_key = project(attn.to_k, hidden_states).unflatten(-1, shape)
            key = torch.cat((prefix_key, normalize_rotate(raw_key, attn.norm_k, rotary_emb)), dim=1)
            value = torch.cat((prefix_value, project(attn.to_v, hidden_states).unflatten(-1, shape)), dim=1)
        query = normalize_rotate(raw_query.unflatten(-1, shape), attn.norm_q, rotary_emb)
        output = dispatch_attention_fn(
            query, key, value,
            attn_mask=attention_mask, dropout_p=0.0, backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        return attn.to_out[1](project(attn.to_out[0], output.flatten(2, 3)))
