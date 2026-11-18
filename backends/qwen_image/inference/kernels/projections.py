import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor

from .activation import swiglu
from .linear import VERIFIED_ROWS, project


@triton.jit
def _project_multiple(inputs, first_weight, second_weight, third_weight, first_out, second_out, third_out,
                      rows, columns: tl.constexpr, inner: tl.constexpr):
    program = tl.program_id(0)
    column_tiles = tl.cdiv(columns, 64)
    first = program // (32 * column_tiles) * 32
    group_rows = tl.minimum(tl.cdiv(rows, 128) - first, 32)
    row = first + program % (32 * column_tiles) % group_rows
    column = program % (32 * column_tiles) // group_rows
    first_acc = tl.full((128, 64), 0, tl.float32)
    second_acc = tl.full((128, 64), 0, tl.float32)
    third_acc = tl.full((128, 64), 0, tl.float32)
    for step in range(inner // 32):
        x = inputs.load([row * 128, step * 32])
        first_tile = first_weight.load([column * 64, step * 32])
        second_tile = second_weight.load([column * 64, step * 32])
        first_acc = tl.dot(x, tl.trans(first_tile), first_acc)
        second_acc = tl.dot(x, tl.trans(second_tile), second_acc)
        if third_weight is not None:
            third_tile = third_weight.load([column * 64, step * 32])
            third_acc = tl.dot(x, tl.trans(third_tile), third_acc)
    offsets = [row * 128, column * 64]
    if third_weight is None:
        gate = second_acc.to(tl.bfloat16).to(tl.float32)
        activation = tl.div_rn(gate, 1 + libdevice.exp(-gate)).to(tl.bfloat16)
        result = activation.to(tl.float32) * first_acc.to(tl.bfloat16).to(tl.float32)
        first_out.store(offsets, result.to(tl.bfloat16))
    else:
        first_out.store(offsets, first_acc.to(tl.bfloat16))
        second_out.store(offsets, second_acc.to(tl.bfloat16))
        third_out.store(offsets, third_acc.to(tl.bfloat16))


def project_swiglu(gate_layer, projection, inputs):
    inner = inputs.shape[-1]
    rows = inputs.numel() // inner
    columns = projection.out_features
    if rows not in VERIFIED_ROWS or (inner, columns) != (4096, 12288) or not inputs.is_contiguous():
        return swiglu(project(gate_layer, inputs), project(projection, inputs))
    result = inputs.new_empty((*inputs.shape[:-1], columns))
    x = TensorDescriptor.from_tensor(inputs.view(rows, inner), [128, 32])
    gate = TensorDescriptor.from_tensor(gate_layer.weight, [64, 32])
    weight = TensorDescriptor.from_tensor(projection.weight, [64, 32])
    output = TensorDescriptor.from_tensor(result.view(rows, columns), [128, 64])
    _project_multiple[(triton.cdiv(rows, 128) * triton.cdiv(columns, 64),)](
        x, weight, gate, None, output, None, None, rows, columns, inner, num_warps=4, num_stages=2,
    )
    return result


def project_qkv(attention, inputs, value):
    rows = inputs.shape[1]
    channels = inputs.shape[2]
    query = inputs.new_empty(inputs.shape)
    key = inputs.new_empty(inputs.shape)
    x = TensorDescriptor.from_tensor(inputs.view(rows, channels), [128, 32])
    weights = [TensorDescriptor.from_tensor(layer.weight, [64, 32])
               for layer in (attention.to_q, attention.to_k, attention.to_v)]
    outputs = [TensorDescriptor.from_tensor(tensor.view(rows, channels), [128, 64])
               for tensor in (query, key, value)]
    _project_multiple[(triton.cdiv(rows, 128) * triton.cdiv(channels, 64),)](
        x, *weights, *outputs, rows, channels, channels, num_warps=4, num_stages=3,
    )
    return query, key
