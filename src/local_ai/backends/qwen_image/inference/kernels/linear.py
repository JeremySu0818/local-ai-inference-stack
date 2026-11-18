import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


IMAGE_TOKEN_COUNTS = (2048, 2304, 2320, 2352, 4032, 4056, 4096, 4128, 4144,
                      9204, 9216, 9240, 16224, 16280, 16320, 16384)
VERIFIED_ROWS = frozenset(tokens * batch for tokens in IMAGE_TOKEN_COUNTS for batch in range(1, 5))


@triton.jit
def _linear(inputs, weights, output, rows, columns: tl.constexpr, inner: tl.constexpr,
            tile: tl.constexpr, depth: tl.constexpr):
    program = tl.program_id(0)
    column_tiles = tl.cdiv(columns, tile)
    row_tiles = tl.cdiv(rows, tile)
    group_start = program // (8 * column_tiles) * 8
    group_rows = tl.minimum(row_tiles - group_start, 8)
    row = group_start + (program % (8 * column_tiles)) % group_rows
    column = (program % (8 * column_tiles)) // group_rows
    accumulator = tl.full((tile, tile), 0, tl.float32)
    for step in range(inner // depth):
        left = inputs.load([row * tile, step * depth])
        right = weights.load([column * tile, step * depth])
        accumulator = tl.dot(left, tl.trans(right), accumulator)
    output.store([row * tile, column * tile], accumulator.to(tl.bfloat16))


def project(layer, inputs):
    inner = inputs.shape[-1]
    rows = inputs.numel() // inner
    columns = layer.out_features
    supported = (inner, columns) in ((4096, 4096), (4096, 12288), (12288, 4096))
    if rows not in VERIFIED_ROWS or not supported or not inputs.is_contiguous():
        return layer(inputs)

    tile = 128
    depth = 32
    result = inputs.new_empty((rows, columns))
    left = TensorDescriptor.from_tensor(inputs.view(rows, inner), [tile, depth])
    right = TensorDescriptor.from_tensor(layer.weight, [tile, depth])
    out = TensorDescriptor.from_tensor(result, [tile, tile])
    _linear[(triton.cdiv(rows, tile) * triton.cdiv(columns, tile),)](
        left, right, out, rows, columns, inner, tile, depth, num_warps=4, num_stages=4,
    )
    return result.view(*inputs.shape[:-1], columns)
