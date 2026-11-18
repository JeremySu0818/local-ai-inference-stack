import torch
import triton
import triton.language as tl


@triton.jit
def _normalize_rotate(source, weight, rotation, output, tokens, heads: tl.constexpr,
                      epsilon: tl.constexpr, rows: tl.constexpr):
    row = tl.program_id(0) * rows + tl.arange(0, rows)
    channel = tl.arange(0, 128)
    offsets = row[:, None] * 128 + channel[None, :]
    value = tl.load(source + offsets).to(tl.float32)
    squares = tl.reshape(value * value, (rows, 32, 2, 2))
    even, odd = tl.split(squares)
    a, c = tl.split(even)
    b, d = tl.split(odd)
    sums = ((a + b) + c) + d
    for i in tl.static_range(5):
        lane = tl.arange(0, 32) ^ (16 >> i)
        sums += tl.gather(sums, tl.broadcast_to(lane[None, :], (rows, 32)), 1)
    inv_rms = tl.rsqrt(tl.max(sums, axis=1) / 128 + epsilon)
    normalized = (value * inv_rms[:, None]).to(tl.bfloat16).to(tl.float32)
    scaled = (normalized * tl.load(weight + channel)[None, :]).to(tl.bfloat16).to(tl.float32)
    real, imag = tl.split(tl.reshape(scaled, (rows, 64, 2)))
    token = row // heads % tokens
    angles = tl.load(rotation + token[:, None] * 128 + channel[None, :])
    cos, sin = tl.split(tl.reshape(angles, (rows, 64, 2)))
    rotated_real = tl.fma(real, cos, -imag * sin)
    rotated_imag = tl.fma(imag, cos, real * sin)
    tl.store(output + offsets, tl.reshape(tl.join(rotated_real, rotated_imag), (rows, 128)))


def normalize_rotate(value, norm, rotation, destination=None):
    result = torch.empty_like(value) if destination is None else destination
    rows = 4
    _normalize_rotate[(value.numel() // (128 * rows),)](
        value, norm.weight, torch.view_as_real(rotation), result, value.shape[1],
        value.shape[2], norm.eps, rows, enable_fp_fusion=False,
    )
    return result
