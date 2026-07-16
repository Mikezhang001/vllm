# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark vllm_mega_moe W4A8 vs the CUTLASS W4A8 backend on SM90."""

import statistics
from dataclasses import dataclass

import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import (
    run_cutlass_moe_w4a8_fp8,
)
from vllm.model_executor.layers.fused_moe.experts.vllm_mega_w4a8_moe import (
    _load_tuned_configs,
    _resolve_kernel_config,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
    MoEPermuteScratch,
)
from vllm.model_executor.layers.fused_moe.oracle.w4a8 import (
    convert_to_w4a8_moe_kernel_format,
    convert_to_w4a8_vllm_mega_moe_kernel_format,
)
from vllm.platforms import current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser

DEV = "cuda:0"
FP8_E4M3_MAX = 448.0
GROUP_SIZE = 128


@dataclass(frozen=True)
class Shape:
    label: str
    M: int
    E: int
    hidden: int
    inter: int
    top_k: int

    @property
    def N(self) -> int:
        return 2 * self.inter


SHAPE_PRESETS: dict[str, list[Shape]] = {
    # inter=128 cases -- CUTLASS rejects (needs %256), mega only.
    "qwen35-tp8": [
        Shape("qwen35-decode-32   ", M=32, E=512, hidden=2048, inter=128, top_k=8),
        Shape("qwen35-decode-64   ", M=64, E=512, hidden=2048, inter=128, top_k=8),
        Shape("qwen35-decode-128  ", M=128, E=512, hidden=2048, inter=128, top_k=8),
        Shape("qwen35-mid-256     ", M=256, E=512, hidden=2048, inter=128, top_k=8),
    ],
    "default": [
        Shape("default-mid-I256   ", M=128, E=512, hidden=2048, inter=256, top_k=8),
        Shape("default-prefill-1k ", M=1024, E=512, hidden=2048, inter=256, top_k=8),
    ],
}


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


def gen_weights(shp: Shape):
    torch.manual_seed(42)
    E, N, K, inter = shp.E, shp.N, shp.hidden, shp.inter
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
    return w13_int8, w2_int8, w13_scale, w2_scale


def gen_inputs(shp: Shape):
    torch.manual_seed(7)
    x = torch.randn(shp.M, shp.hidden, dtype=torch.bfloat16, device=DEV) * 0.1
    logits = torch.randn(shp.M, shp.E, dtype=torch.float32, device=DEV)
    topk_weights, topk_ids = torch.topk(
        torch.softmax(logits, dim=-1), shp.top_k, dim=-1
    )
    topk_weights = (topk_weights / topk_weights.sum(-1, keepdim=True)).contiguous()
    topk_ids = topk_ids.to(torch.int32).contiguous()
    return x, topk_ids, topk_weights


def prep_mega(shp, w13_int8, w2_int8, w13_scale, w2_scale):
    w13_p = _pack_uint4b8_int32(w13_int8).contiguous()
    w2_p = _pack_uint4b8_int32(w2_int8).contiguous()
    return convert_to_w4a8_vllm_mega_moe_kernel_format(
        w13_weight_packed=w13_p,
        w2_weight_packed=w2_p,
        w13_weight_scale=w13_scale.to(torch.bfloat16),
        w2_weight_scale=w2_scale.to(torch.bfloat16),
    )


def prep_cutlass(shp, w13_int8, w2_int8, w13_scale, w2_scale):
    E, K, inter = shp.E, shp.hidden, shp.inter
    w13_p = _pack_uint4b8_int32(w13_int8).contiguous()
    w2_p = _pack_uint4b8_int32(w2_int8).contiguous()
    (w13, w2, w13sc, w2sc, w13chan, w2chan, bstr1, bstr2) = (
        convert_to_w4a8_moe_kernel_format(
            w13_weight_packed=w13_p,
            w2_weight_packed=w2_p,
            w13_weight_scale=w13_scale.to(torch.bfloat16),
            w2_weight_scale=w2_scale.to(torch.bfloat16),
        )
    )
    a_str1 = torch.full((E,), K, device=DEV, dtype=torch.int64)
    a_str2 = torch.full((E,), inter, device=DEV, dtype=torch.int64)
    c_str1 = torch.full((E,), 2 * inter, device=DEV, dtype=torch.int64)
    c_str2 = a_str1
    s_str1 = torch.zeros((E, 2), device=DEV, dtype=torch.int64)
    s_str1[:, 0] = 2 * inter
    s_str2 = torch.zeros((E, 2), device=DEV, dtype=torch.int64)
    s_str2[:, 0] = K
    return {
        "w13": w13,
        "w2": w2,
        "w13_scale": w13sc,
        "w2_scale": w2sc,
        "w13_chan": w13chan,
        "w2_chan": w2chan,
        "a_strides1": a_str1,
        "a_strides2": a_str2,
        "b_strides1": bstr1,
        "b_strides2": bstr2,
        "c_strides1": c_str1,
        "c_strides2": c_str2,
        "s_strides1": s_str1,
        "s_strides2": s_str2,
    }


@dataclass
class Buffers:
    x_fp8: torch.Tensor
    x_scale: torch.Tensor
    mega_out: torch.Tensor
    cutlass_out: torch.Tensor
    ws13: torch.Tensor
    ws2: torch.Tensor
    permute_scratch: MoEPermuteScratch


def alloc_buffers(shp: Shape) -> Buffers:
    M, K, inter, topk, E = shp.M, shp.hidden, shp.inter, shp.top_k, shp.E
    N = 2 * inter
    return Buffers(
        x_fp8=torch.empty(M, K, dtype=torch.float8_e4m3fn, device=DEV),
        x_scale=torch.empty(M, 1, dtype=torch.float32, device=DEV),
        mega_out=torch.empty(M, K, dtype=torch.bfloat16, device=DEV),
        cutlass_out=torch.empty(M, K, dtype=torch.bfloat16, device=DEV),
        ws13=torch.empty(M * topk, max(N, K), dtype=torch.bfloat16, device=DEV),
        ws2=torch.empty(M * topk, max(inter, K), dtype=torch.bfloat16, device=DEV),
        permute_scratch=MoEPermuteScratch(
            max_num_tokens=M,
            topk=topk,
            num_experts=E,
            num_local_experts=E,
            device=torch.device(DEV),
            hidden_size=K,
            hidden_dtype=torch.float8_e4m3fn,
        ),
    )


def run_mega(shp, x_bf16, topk_ids, topk_weights, mega_w, cfg, buf):
    w13_pack, w2_pack, w13_sc, w2_sc = mega_w
    block_m, block_n, warp_n, stages = cfg
    torch.ops._C.dynamic_per_token_scaled_fp8_quant(
        buf.x_fp8, x_bf16, buf.x_scale, None
    )
    x_scale_1d = buf.x_scale.view(-1)
    sorted_ids, expert_ids, num_pad = moe_align_block_size(
        topk_ids, block_m, shp.E, expert_map=None, ignore_invalid_experts=True
    )
    buf.mega_out.zero_()
    torch.ops._moe_C.vllm_mega_moe_fused_w4a8_up_down(
        buf.x_fp8,
        x_scale_1d,
        w13_pack,
        w13_sc,
        w2_pack,
        w2_sc,
        sorted_ids,
        expert_ids,
        num_pad,
        topk_weights,
        buf.mega_out,
        shp.top_k,
        block_m,
        block_n,
        warp_n,
        stages,
        1.0,
    )
    return buf.mega_out


def run_cutlass(shp, x_bf16, topk_ids, topk_weights, cw, buf):
    torch.ops._C.dynamic_per_token_scaled_fp8_quant(
        buf.x_fp8, x_bf16, buf.x_scale, None
    )
    run_cutlass_moe_w4a8_fp8(
        output=buf.cutlass_out,
        hidden_states=buf.x_fp8,
        w1=cw["w13"],
        w2=cw["w2"],
        topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=shp.E,
        expert_map=None,
        w1_scale=cw["w13_scale"],
        w2_scale=cw["w2_scale"],
        a1q_scale=buf.x_scale,
        a2_scale=None,
        w1_chan_scale=cw["w13_chan"],
        w2_chan_scale=cw["w2_chan"],
        a_strides1=cw["a_strides1"],
        a_strides2=cw["a_strides2"],
        b_strides1=cw["b_strides1"],
        b_strides2=cw["b_strides2"],
        c_strides1=cw["c_strides1"],
        c_strides2=cw["c_strides2"],
        s_strides1=cw["s_strides1"],
        s_strides2=cw["s_strides2"],
        workspace13=buf.ws13,
        workspace2=buf.ws2,
        expert_num_tokens=None,
        out_dtype=torch.bfloat16,
        per_act_token=True,
        per_out_ch=True,
        use_batched_format=False,
        topk_weights=topk_weights,
        group_size=GROUP_SIZE,
        permute_scratch=buf.permute_scratch,
    )
    return buf.cutlass_out


def time_fn(fn, warmup: int, iters: int) -> float:
    """Return median wall-clock latency in microseconds."""
    for _ in range(warmup):
        _ = fn()
    torch.accelerator.synchronize()
    starts = [torch.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        _ = fn()
        ends[i].record()
    torch.accelerator.synchronize()
    return statistics.median(s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends))


def bench_shape(shp: Shape, warmup: int, iters: int):
    w13_int8, w2_int8, w13_sc, w2_sc = gen_weights(shp)
    mega_w = prep_mega(shp, w13_int8, w2_int8, w13_sc, w2_sc)
    x, topk_ids, topk_weights = gen_inputs(shp)
    cfg = _resolve_kernel_config(shp.M, shp.E, shp.N)
    buf = alloc_buffers(shp)

    # Warm mega numerics + time it.
    _ = run_mega(shp, x, topk_ids, topk_weights, mega_w, cfg, buf)
    t_mega = time_fn(
        lambda: run_mega(shp, x, topk_ids, topk_weights, mega_w, cfg, buf),
        warmup=warmup,
        iters=iters,
    )

    # CUTLASS only if intermediate size is a multiple of 256.
    if shp.inter % 256 != 0:
        return t_mega, None, None, cfg

    cw = prep_cutlass(shp, w13_int8, w2_int8, w13_sc, w2_sc)
    o_mega = run_mega(shp, x, topk_ids, topk_weights, mega_w, cfg, buf)
    o_cutlass = run_cutlass(shp, x, topk_ids, topk_weights, cw, buf)
    cos = torch.nn.functional.cosine_similarity(
        o_mega.flatten().float().unsqueeze(0),
        o_cutlass.flatten().float().unsqueeze(0),
    ).item()
    t_cutlass = time_fn(
        lambda: run_cutlass(shp, x, topk_ids, topk_weights, cw, buf),
        warmup=warmup,
        iters=iters,
    )
    return t_mega, t_cutlass, cos, cfg


def main(args):
    shapes: list[Shape] = []
    for preset in args.shapes:
        if preset == "all":
            for s in SHAPE_PRESETS.values():
                shapes.extend(s)
        else:
            shapes.extend(SHAPE_PRESETS[preset])

    # QuantFP8 (used inside the CUTLASS weight prep) reads vLLM's global
    # config; wrap the whole benchmark in a default VllmConfig context.
    from vllm.config import VllmConfig, set_current_vllm_config

    device_name = (
        current_platform.get_device_name(0).replace(" ", "_").replace("/", "_")
    )
    print(f"# device: {device_name}")
    print(f"# warmup={args.warmup}, iters={args.iters}, median latency (us)")
    print(
        f"{'shape':<22s} {'M':>5s} {'E':>4s} {'K':>5s} {'I':>4s} {'topk':>4s} "
        f"{'tuned':>6s} {'mega_us':>9s} {'cutlass_us':>11s} "
        f"{'speedup':>8s} {'cos':>7s}"
    )
    print("-" * 108)
    with set_current_vllm_config(VllmConfig()):
        for shp in shapes:
            tuned_hit = _load_tuned_configs(shp.E, shp.N, device_name) is not None
            t_mega, t_cutlass, cos, cfg = bench_shape(shp, args.warmup, args.iters)
            if t_cutlass is None:
                print(
                    f"{shp.label:<22s} {shp.M:>5d} {shp.E:>4d} "
                    f"{shp.hidden:>5d} {shp.inter:>4d} {shp.top_k:>4d} "
                    f"{'yes' if tuned_hit else 'no ':>6s} "
                    f"{t_mega:>9.1f} {'N/A':>11s} "
                    f"{'CUTLASS I%256':>8s} {'-':>7s}"
                )
            else:
                spd = t_cutlass / t_mega
                print(
                    f"{shp.label:<22s} {shp.M:>5d} {shp.E:>4d} "
                    f"{shp.hidden:>5d} {shp.inter:>4d} {shp.top_k:>4d} "
                    f"{'yes' if tuned_hit else 'no ':>6s} "
                    f"{t_mega:>9.1f} {t_cutlass:>11.1f} "
                    f"{spd:>7.2f}x {cos:>7.4f}"
                )


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark vllm_mega_moe W4A8 vs CUTLASS W4A8"
    )
    parser.add_argument(
        "--shapes",
        nargs="+",
        type=str,
        default=["qwen35-tp8", "default"],
        choices=list(SHAPE_PRESETS.keys()) + ["all"],
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    main(parser.parse_args())
