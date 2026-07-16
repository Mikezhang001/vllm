# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import Enum
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    int4_w4afp8_moe_quant_config,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8DynamicTokenSym,
    kInt4Static,
)

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)


class W4A8MoeBackend(Enum):
    CUTLASS = "CUTLASS"
    VLLM_MEGA_MOE = "VLLM_MEGA_MOE"


def backend_to_kernel_cls(
    backend: W4A8MoeBackend,
) -> list[type[mk.FusedMoEExpertsModular]]:
    if backend == W4A8MoeBackend.CUTLASS:
        from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import (
            CutlassExpertsW4A8Fp8,
        )

        return [CutlassExpertsW4A8Fp8]
    if backend == W4A8MoeBackend.VLLM_MEGA_MOE:
        from vllm.model_executor.layers.fused_moe.experts.vllm_mega_w4a8_moe import (
            VllmMegaExpertsW4A8Fp8,
        )

        return [VllmMegaExpertsW4A8Fp8]
    raise ValueError(f"Unknown W4A8 MoE backend: {backend.value}")


_ENV_TO_BACKEND: dict[str, W4A8MoeBackend] = {
    "cutlass": W4A8MoeBackend.CUTLASS,
    "vllm_mega_moe": W4A8MoeBackend.VLLM_MEGA_MOE,
}


def select_w4a8_moe_backend(
    config: FusedMoEConfig,
    weight_key: QuantKey | None = kInt4Static,
    activation_key: QuantKey | None = kFp8DynamicTokenSym,
) -> tuple[W4A8MoeBackend, type[mk.FusedMoEExpertsModular]]:
    # ``VLLM_W4A8_MOE_BACKEND`` picks between the in-tree CUTLASS kernel and
    # ``vllm_mega_moe``, an in-tree one-kernel-fused WGMMA kernel that
    # handles smaller intermediate_size_per_partition values (e.g. 128)
    # which the CUTLASS backend rejects due to its %256 tile constraint.
    backend = _ENV_TO_BACKEND[envs.VLLM_W4A8_MOE_BACKEND]

    activation_format = (
        mk.FusedMoEActivationFormat.BatchedExperts
        if config.moe_parallel_config.use_batched_activation_format
        else mk.FusedMoEActivationFormat.Standard
    )

    last_reason: str | None = None
    for kernel_cls in backend_to_kernel_cls(backend):
        supported, reason = kernel_cls.is_supported_config(
            kernel_cls,
            config,
            weight_key,
            activation_key,
            activation_format,
        )
        if supported:
            logger.info_once("Using %s W4A8 MoE backend.", backend.value)
            return backend, kernel_cls
        last_reason = reason

    raise NotImplementedError(
        f"W4A8 MoE backend {backend.value} does not support the "
        f"deployment configuration: {last_reason}."
    )


def convert_to_w4a8_moe_kernel_format(
    w13_weight_packed: torch.Tensor,
    w2_weight_packed: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight_scale: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        GroupShape,
        convert_bf16_scales_to_fp8,
        convert_packed_uint4b8_to_signed_int4_inplace,
    )

    quant_fp8 = QuantFP8(static=False, group_shape=GroupShape.PER_TOKEN)

    convert_packed_uint4b8_to_signed_int4_inplace(w13_weight_packed)
    # Mirror the sync in CutlassW4A8LinearKernel; required for TP>1 correctness.
    torch.accelerator.synchronize()
    w13_weight_shuffled, b_strides1 = ops.cutlass_encode_and_reorder_int4b_grouped(
        w13_weight_packed
    )

    convert_packed_uint4b8_to_signed_int4_inplace(w2_weight_packed)
    # Mirror the sync in CutlassW4A8LinearKernel; required for TP>1 correctness.
    torch.accelerator.synchronize()
    w2_weight_shuffled, b_strides2 = ops.cutlass_encode_and_reorder_int4b_grouped(
        w2_weight_packed
    )

    w13_weight_scale, w13_weight_chan_scale = convert_bf16_scales_to_fp8(
        quant_fp8, w13_weight_scale
    )
    w2_weight_scale, w2_weight_chan_scale = convert_bf16_scales_to_fp8(
        quant_fp8, w2_weight_scale
    )

    # Scales are stored as (E, N, K // 128), but the kernel expects
    # (E, K // 128, N) in row-major format.
    w13_weight_scale_packed = ops.cutlass_pack_scale_fp8(
        w13_weight_scale.permute(0, 2, 1).contiguous()
    )
    w2_weight_scale_packed = ops.cutlass_pack_scale_fp8(
        w2_weight_scale.permute(0, 2, 1).contiguous()
    )

    return (
        w13_weight_shuffled,
        w2_weight_shuffled,
        w13_weight_scale_packed,
        w2_weight_scale_packed,
        w13_weight_chan_scale,
        w2_weight_chan_scale,
        b_strides1,
        b_strides2,
    )


def convert_to_w4a8_vllm_mega_moe_kernel_format(
    w13_weight_packed: torch.Tensor,
    w2_weight_packed: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert vLLM standard W4A8 weight layout to what the vllm_mega_moe
    WGMMA kernel expects.

    The kernel expects:
      * ``w13`` as ``uint8 [E, 2*I, K/2]`` with each byte holding two
        packed INT4 nibbles (low = even K index, high = odd K index)
        and the ``2*I`` axis interleaved rep=8 (every 8 gate rows are
        followed by 8 up rows for in-kernel SwiGLU).
      * ``w13_scale`` as ``fp32 [E, 2*I, K/128]`` with the same rep=8
        interleave on the ``2*I`` axis.
      * ``w2`` as ``uint8 [E, K, I/2]`` (no interleave).
      * ``w2_scale`` as ``fp32 [E, K, I/128]`` (no interleave).

    Inputs are the vLLM standard ``[E, 2*I, K/8] int32`` (uint4b8
    packed) weights and ``[E, 2*I, K/128] bf16`` group scales, i.e. the
    parameters that :class:`CompressedTensorsW4A8Fp8MoEMethod.create_
    weights` registers before this transform runs.
    """
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        convert_packed_uint4b8_to_signed_int4_inplace,
    )

    convert_packed_uint4b8_to_signed_int4_inplace(w13_weight_packed)
    convert_packed_uint4b8_to_signed_int4_inplace(w2_weight_packed)
    # Mirror the sync used in the CUTLASS conversion path; required for
    # TP>1 correctness (see CutlassW4A8LinearKernel comments).
    torch.accelerator.synchronize()

    # int32 [E, N, K/8] == bytewise uint8 [E, N, K/2] with the same
    # nibble order the kernel expects.
    w13_u8 = w13_weight_packed.view(torch.uint8).contiguous()
    w2_u8 = w2_weight_packed.view(torch.uint8).contiguous()

    # rep=8 interleave on w13's N axis: vLLM checkpoints store gate || up
    # concatenated along N; the kernel expects blocks of 8 gate rows
    # followed by 8 up rows so it can fuse SwiGLU inside the kernel.
    w13_u8_interleaved = _interleave_gate_up_rep8(w13_u8)
    w13_scale_f32 = w13_weight_scale.float().contiguous()
    w13_scale_interleaved = _interleave_gate_up_rep8(w13_scale_f32)

    w2_scale_f32 = w2_weight_scale.float().contiguous()

    return (
        w13_u8_interleaved,
        w2_u8,
        w13_scale_interleaved,
        w2_scale_f32,
    )


def _interleave_gate_up_rep8(t: torch.Tensor) -> torch.Tensor:
    """Interleave gate/up rows on axis 1 with block size 8.

    Given a tensor with shape ``[E, 2*I, ...]`` where axis 1 is
    ``[gate_rows(I), up_rows(I)]``, returns a tensor of the same shape
    where axis 1 is ``[g0..g7, u0..u7, g8..g15, u8..u15, ...]``.
    """
    e, n_full, *tail = t.shape
    assert n_full % 16 == 0, (
        f"gate/up interleave requires 2*I divisible by 16 (rep=8 * 2 "
        f"groups), got 2*I={n_full}"
    )
    i = n_full // 2
    assert i % 8 == 0, f"I must be divisible by 8 for rep=8 interleave, got I={i}"
    gate = t[:, :i]
    up = t[:, i:]
    # Reshape to expose the rep=8 chunks, stack, and flatten back.
    gate_blocks = gate.reshape(e, i // 8, 8, *tail)
    up_blocks = up.reshape(e, i // 8, 8, *tail)
    interleaved = torch.stack([gate_blocks, up_blocks], dim=2)
    return interleaved.reshape(e, n_full, *tail).contiguous()


def make_w4a8_moe_quant_config(
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    g1_alphas: torch.Tensor,
    g2_alphas: torch.Tensor,
) -> FusedMoEQuantConfig:
    return int4_w4afp8_moe_quant_config(
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        g1_alphas=g1_alphas,
        g2_alphas=g2_alphas,
        per_act_token_quant=True,
        per_out_ch_quant=True,
    )


def make_w4a8_moe_kernel(
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    experts_cls: type[mk.FusedMoEExpertsModular],
    b_strides1: torch.Tensor | None,
    b_strides2: torch.Tensor | None,
    group_size: int,
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> mk.FusedMoEKernel:
    prepare_finalize = maybe_make_prepare_finalize(
        moe=moe_config,
        quant_config=moe_quant_config,
        routing_tables=routing_tables,
        allow_new_interface=True,
    )
    assert prepare_finalize is not None

    logger.info_once("Using %s", prepare_finalize.__class__.__name__)

    # ``b_strides1``/``b_strides2`` come from the CUTLASS reorder path and
    # are only meaningful for :class:`CutlassExpertsW4A8Fp8`. Alternative
    # W4A8 experts backends (e.g. ``VllmMegaExpertsW4A8Fp8``) do not
    # need per-expert strides.
    from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import (
        CutlassExpertsW4A8Fp8,
    )

    if issubclass(experts_cls, CutlassExpertsW4A8Fp8):
        assert b_strides1 is not None and b_strides2 is not None
        experts: mk.FusedMoEExpertsModular = experts_cls(
            moe_config=moe_config,
            quant_config=moe_quant_config,
            b_strides1=b_strides1,
            b_strides2=b_strides2,
            group_size=group_size,
        )
    else:
        experts = experts_cls(
            moe_config=moe_config,
            quant_config=moe_quant_config,
        )

    return mk.FusedMoEKernel(
        prepare_finalize,
        experts,
    )
