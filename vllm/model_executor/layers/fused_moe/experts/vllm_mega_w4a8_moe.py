# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W4A8 MoE experts backed by an in-tree WGMMA one-kernel-fused kernel."""

from __future__ import annotations

import functools
import json
import os

import torch

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8DynamicTokenSym,
    kInt4Static,
)

logger = init_logger(__name__)


def _kernel_available() -> bool:
    moe_c = getattr(torch.ops, "_moe_C", None)
    if moe_c is None:
        return False
    return hasattr(moe_c, "vllm_mega_moe_fused_w4a8_up_down")


_DEFAULT_BLOCK_M = 16
_DEFAULT_BLOCK_N = 32
_DEFAULT_WARP_N = 8
_DEFAULT_STAGES = 3


def _pick_block_m(m: int) -> int:
    for candidate in (8, 16, 32, 48, 64, 128):
        if m <= candidate * 4:
            return candidate
    return 128


@functools.lru_cache(maxsize=64)
def _load_tuned_configs(
    e: int, n_full: int, device_name: str
) -> dict[int, dict[str, int]] | None:
    fname = (
        f"E={e},N={n_full},device_name={device_name}"
        f",dtype=w4a8_fp8,backend=vllm_mega_moe.json"
    )
    candidates: list[str] = []
    if envs.VLLM_TUNED_CONFIG_FOLDER is not None:
        candidates.append(os.path.join(envs.VLLM_TUNED_CONFIG_FOLDER, fname))
    configs_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "configs"
    )
    candidates.append(os.path.join(configs_dir, fname))

    for path in candidates:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            raw = json.load(f)
        logger.info_once("Using vllm_mega_moe W4A8 tuning table from %s", path)
        return {int(k): v for k, v in raw.items()}

    logger.warning_once(
        "No vllm_mega_moe W4A8 tuning table found (searched %s); using "
        "runtime heuristic.",
        ", ".join(candidates),
    )
    return None


def _resolve_kernel_config(m: int, e: int, n_full: int) -> tuple[int, int, int, int]:
    device_name = torch.cuda.get_device_name(0).replace(" ", "_").replace("/", "_")
    tuned = _load_tuned_configs(e, n_full, device_name)
    if tuned is not None:
        best_m = min(tuned.keys(), key=lambda k: abs(k - m))
        row = tuned[best_m]
        return (
            int(row["block_m"]),
            int(row["block_n"]),
            int(row["warp_n"]),
            int(row["stages"]),
        )
    return (_pick_block_m(m), _DEFAULT_BLOCK_N, _DEFAULT_WARP_N, _DEFAULT_STAGES)


class VllmMegaExpertsW4A8Fp8(mk.FusedMoEExpertsModular):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        self.out_dtype = moe_config.in_dtype

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        if moe_config.in_dtype != torch.bfloat16:
            return (
                False,
                f"kernel does not support {moe_config.in_dtype} input/output dtype",
            )
        if moe_config.hidden_dim % 256 != 0:
            return (
                False,
                f"hidden_size ({moe_config.hidden_dim}) must be a multiple of 256",
            )
        return mk.FusedMoEExperts.is_supported_config(
            cls,
            moe_config,
            weight_key,
            activation_key,
            activation_format,
        )

    @staticmethod
    def _supports_current_device() -> bool:
        if not _kernel_available():
            return False
        if not torch.cuda.is_available():
            return False
        major, _ = torch.cuda.get_device_capability()
        return major == 9

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kInt4Static, kFp8DynamicTokenSym)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return not moe_parallel_config.use_batched_activation_format

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # Kernel writes topk-reduced [M, K] output directly; no scratch needed.
        return (0,), (0,), (M, K)

    def workspace_dtype(self, act_dtype: torch.dtype) -> torch.dtype:
        return self.out_dtype

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        assert activation == MoEActivation.SILU
        assert hidden_states.dtype == torch.float8_e4m3fn
        assert a1q_scale is not None
        assert w1.dtype == torch.uint8 and w2.dtype == torch.uint8
        assert self.w1_scale is not None and self.w2_scale is not None
        assert expert_map is None, "expert_map (EP) not supported yet"

        m = hidden_states.size(0)
        e = w1.size(0)
        n_full = w1.size(1)
        top_k = topk_ids.size(1)

        block_m, block_n, warp_n, stages = _resolve_kernel_config(m, e, n_full)

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids,
            block_m,
            e,
            expert_map,
            ignore_invalid_experts=True,
        )

        output.zero_()

        torch.ops._moe_C.vllm_mega_moe_fused_w4a8_up_down(
            hidden_states,
            a1q_scale,
            w1,
            self.w1_scale,
            w2,
            self.w2_scale,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            topk_weights.to(torch.float32),
            output,
            top_k,
            block_m,
            block_n,
            warp_n,
            stages,
            1.0,
        )
