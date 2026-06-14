# Triton kernels

The detailed Triton implementation note lives next to the kernel source: [`../flowmatching_lthc/models/kernel_note.md`](../flowmatching_lthc/models/kernel_note.md).

## B/4 split-output all-base read

`jit_thc::all_base_b4_12_split_traceable` is the fused base-read boundary used
by the non-shared read/write dense models.

Input:

- `x0`: `[B, 64, 64, C]`
- `alpha_stack`: `[12, C, 16]`

Output:

- twelve separate tensors `z0 ... z11`, each `[B, 256, C]`

This differs from the older `all_base_b4_12_traceable`, which returns one
stacked `[12, B, 256, C]` tensor. The separate-output form avoids a stack/slice
boundary in the consumer graph. Its backward stacks the incoming 12 gradients
once, reuses the fused `grad_x` kernel, and computes `grad_alpha` with one
streaming pass over `x0` and the grad stack.

Default tuning knobs:

- `JIT_THC_READ12_SPLIT_BLOCK_M=2`
- `JIT_THC_READ12_SPLIT_BLOCK_C=64`
- `JIT_THC_READ12_SPLIT_WARPS=2`
- `JIT_THC_READ12_SPLIT_GRAD_ALPHA_ROWS=64`
- `JIT_THC_READ12_SPLIT_GRAD_ALPHA_BLOCK_C=64`
- `JIT_THC_READ12_SPLIT_GRAD_ALPHA_WARPS=8`

Numerical parity against the older stack all-base op on GPU2:

| dtype | output max abs | grad_x max abs | grad_alpha max abs |
|---|---:|---:|---:|
| fp32 | 0 | 0 | 2.10e-5 |
| bf16 | 0 | 0 | 6.10e-5 |

The `grad_alpha` difference is from fp32 atomic accumulation order.
