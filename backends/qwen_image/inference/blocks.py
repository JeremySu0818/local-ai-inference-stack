from torch import nn

from .attention import CachedAttention
from .kernels.projections import project_swiglu
from .kernels.linear import project


class FeedForward(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.gate_layer = original.gate_layer
        self.proj = original.proj
        self.out = original.out

    def forward(self, hidden_states):
        projected = project_swiglu(self.gate_layer, self.proj, hidden_states)
        return project(self.out, projected)


def optimize_blocks(transformer):
    for block in transformer.transformer_blocks:
        block.attn.set_processor(CachedAttention())
        block.img_mlp = FeedForward(block.img_mlp)
