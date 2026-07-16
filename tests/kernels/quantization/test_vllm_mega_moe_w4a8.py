# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the vllm_mega_moe W4A8 kernel (SM90 only)."""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.oracle.w4a8 import (
    convert_to_w4a8_vllm_mega_moe_kernel_format,
)
from vllm.platforms import current_platform

IS_SUPPORTED_BY_GPU = (
    current_platform.is_cuda()
    and current_platform.get_device_capability()[0] == 9
    and hasattr(torch.ops, "_moe_C")
    and hasattr(torch.ops._moe_C, "vllm_mega_moe_fused_w4a8_up_down")
)

FP8_E4M3_MAX = 448.0
DEV = "cuda:0"
GROUP_SIZE = 128

# (M, E, hidden, I_per_partition, top_k)
TEST_SHAPES = [
    (32, 4, 256, 128, 2),
    (64, 8, 512, 256, 4),
    (64, 16, 2048, 128, 4),
]

# (block_m, block_n, warp_n, stages)
TEST_CFGS = [
    (16, 32, 8, 4),
    (8, 32, 8, 3),
]


def _pack_uint4b8_int32(w_int8: torch.Tensor) -> torch.Tensor:
    assert w_int8.shape[-1] % 8 == 0
    offset = (w_int8.to(torch.int32) + 8) & 0x0F
    Kg = w_int8.shape[-1]
    packed = torch.zeros(
        *w_int8.shape[:-1], Kg // 8, dtype=torch.int32, device=w_int8.device
    )
    for i in range(8):
        packed |= offset[..., i::8] << (i * 4)
    return packed


def _reference_moe_w4a8_fp32(
    x_bf16, w13_int8, w13_scale, w2_int8, w2_scale, topk_ids, topk_weights, top_k
):
    M, K = x_bf16.shape
    E, N_full, _ = w13_int8.shape
    inter = N_full // 2

    x_f = x_bf16.float()
    amax = x_f.abs().amax(dim=-1).clamp(min=1e-12)
    s = amax / FP8_E4M3_MAX
    x_fp8 = (x_f / s.unsqueeze(-1)).to(torch.float8_e4m3fn)
    x_dq_all = x_fp8.float() * s.unsqueeze(-1)

    out = torch.zeros(M, K, dtype=torch.float32, device=x_bf16.device)
    for t in range(M):
        for k in range(top_k):
            eid = int(topk_ids[t, k].item())
            w = float(topk_weights[t, k].item())
            w1_f = w13_int8[eid].float().reshape(N_full, K // GROUP_SIZE, GROUP_SIZE)
            w1_dq = (w1_f * w13_scale[eid].unsqueeze(-1)).reshape(N_full, K)
            y = w1_dq @ x_dq_all[t]
            z = torch.nn.functional.silu(y[:inter]) * y[inter:]
            amax2 = z.abs().amax().clamp(min=1e-12)
            s2 = amax2 / FP8_E4M3_MAX
            z_dq = (z / s2).to(torch.float8_e4m3fn).float() * s2
            w2_f = w2_int8[eid].float().reshape(K, inter // GROUP_SIZE, GROUP_SIZE)
            w2_dq = (w2_f * w2_scale[eid].unsqueeze(-1)).reshape(K, inter)
            out[t] += w * (w2_dq @ z_dq)
    return out.to(torch.bfloat16)


@pytest.mark.skipif(
    not IS_SUPPORTED_BY_GPU,
    reason="vllm_mega_moe requires SM90 with the in-tree kernel built.",
)
@pytest.mark.parametrize("shape", TEST_SHAPES)
@pytest.mark.parametrize("cfg", TEST_CFGS)
def test_vllm_mega_moe_w4a8_matches_fp32_reference(shape, cfg):
    M, E, hidden, inter, top_k = shape
    block_m, block_n, warp_n, stages = cfg
    K = hidden
    N = 2 * inter

    torch.manual_seed(42)
    w13_int8 = torch.randint(-7, 8, (E, N, K), dtype=torch.int8, device=DEV)
    w2_int8 = torch.randint(-7, 8, (E, K, inter), dtype=torch.int8, device=DEV)
    w13_scale = (
        torch.rand((E, N, K // GROUP_SIZE), dtype=torch.float32, device=DEV) * 0.01
        + 0.001
    )
    w2_scale = (
        torch.rand((E, K, inter // GROUP_SIZE), dtype=torch.float32, device=DEV) * 0.01
        + 0.001
    )

    w13_p = _pack_uint4b8_int32(w13_int8).contiguous()
    w2_p = _pack_uint4b8_int32(w2_int8).contiguous()
    w13_pack, w2_pack, w13_sc, w2_sc = convert_to_w4a8_vllm_mega_moe_kernel_format(
        w13_weight_packed=w13_p,
        w2_weight_packed=w2_p,
        w13_weight_scale=w13_scale.to(torch.bfloat16),
        w2_weight_scale=w2_scale.to(torch.bfloat16),
    )

    torch.manual_seed(7)
    x = torch.randn(M, K, dtype=torch.bfloat16, device=DEV) * 0.1
    logits = torch.randn(M, E, dtype=torch.float32, device=DEV)
    topk_weights, topk_ids = torch.topk(torch.softmax(logits, dim=-1), top_k, dim=-1)
    topk_weights = (topk_weights / topk_weights.sum(-1, keepdim=True)).contiguous()
    topk_ids = topk_ids.to(torch.int32).contiguous()

    x_fp8 = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=DEV)
    x_scale = torch.empty(M, 1, dtype=torch.float32, device=DEV)
    torch.ops._C.dynamic_per_token_scaled_fp8_quant(x_fp8, x, x_scale, None)
    x_scale = x_scale.view(-1).contiguous()

    sorted_ids, expert_ids, num_pad = moe_align_block_size(
        topk_ids, block_m, E, expert_map=None, ignore_invalid_experts=True
    )

    out_kernel = torch.zeros_like(x)
    torch.ops._moe_C.vllm_mega_moe_fused_w4a8_up_down(
        x_fp8,
        x_scale,
        w13_pack,
        w13_sc,
        w2_pack,
        w2_sc,
        sorted_ids,
        expert_ids,
        num_pad,
        topk_weights,
        out_kernel,
        top_k,
        block_m,
        block_n,
        warp_n,
        stages,
        1.0,
    )
    torch.accelerator.synchronize()

    out_ref = _reference_moe_w4a8_fp32(
        x, w13_int8, w13_scale, w2_int8, w2_scale, topk_ids, topk_weights, top_k
    )

    a = out_kernel.float().flatten()
    b = out_ref.float().flatten()
    cos = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    max_d = (a - b).abs().max().item()

    assert cos > 0.99, f"cosine={cos:.4f} below threshold for shape={shape} cfg={cfg}"
    assert max_d < 0.5, (
        f"max_diff={max_d:.4f} above threshold for shape={shape} cfg={cfg}"
    )
