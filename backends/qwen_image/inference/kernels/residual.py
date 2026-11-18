import torch
import triton
import triton.language as tl


@triton.jit
def _gated_add(base, update, gate, output, count, tokens, channels: tl.constexpr, block: tl.constexpr):
    index = tl.program_id(0) * block + tl.arange(0, block)
    valid = index < count
    sample = index // (tokens * channels)
    gate_value = tl.load(gate + sample * channels + index % channels, valid, 0)
    gated = (gate_value.to(tl.float32) * tl.load(update + index, valid, 0).to(tl.float32)).to(tl.bfloat16)
    result = tl.load(base + index, valid, 0).to(tl.float32) + gated.to(tl.float32)
    tl.store(output + index, result, valid)


def gated_add(base, update, gate):
    result = torch.empty_like(update)
    count = update.numel()
    _gated_add[(triton.cdiv(count, 512),)](
        base, update, gate, result, count, base.shape[1], base.shape[2], 512, enable_fp_fusion=False,
    )
    return result
