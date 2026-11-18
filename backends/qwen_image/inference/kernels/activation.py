import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _swiglu(gate, projection, output, count, block: tl.constexpr):
    index = tl.program_id(0) * block + tl.arange(0, block)
    valid = index < count
    value = tl.load(gate + index, valid, 0).to(tl.float32)
    activated = tl.div_rn(value, 1 + libdevice.exp(-value)).to(tl.bfloat16)
    projected = tl.load(projection + index, valid, 0)
    tl.store(output + index, activated.to(tl.float32) * projected.to(tl.float32), valid)


def swiglu(gate, projection):
    result = torch.empty_like(gate)
    count = gate.numel()
    _swiglu[(triton.cdiv(count, 512),)](gate, projection, result, count, 512, enable_fp_fusion=False)
    return result
