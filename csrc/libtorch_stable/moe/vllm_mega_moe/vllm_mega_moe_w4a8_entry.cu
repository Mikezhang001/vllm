// SPDX-License-Identifier: Apache-2.0
// Stable-ABI wrapper for the vllm_mega_moe W4A8 WGMMA kernel.
// The device kernel lives in vllm_mega_moe_w4a8_kernel.cu.

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "core/registration.h"
#include "libtorch_stable/torch_utils.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

// Forward decls -- struct + host dispatcher defined in
// vllm_mega_moe_w4a8_kernel.cu.
struct FusedMoeW4A8Args {
  __nv_fp8_e4m3 const* x;
  float const* x_scale;
  uint8_t* w;
  float const* w_scale;
  uint8_t* w2;
  float const* w2_scale;
  __nv_bfloat16* out;
  int const* sorted_token_ids;
  int const* expert_ids;
  int const* num_tokens_post_padded;
  float const* topk_weights;
  int top_k;
  int M, K, N;
  int num_experts;
  int sorted_num;
  int block_m;
  int block_n;
  int warp_n;
  int stages;
  float scaling_factor;
  cudaStream_t stream;
};

void fused_moe_w4a8_wgmma_up_down_acc(FusedMoeW4A8Args const& a);

namespace {

using torch::headeronly::ScalarType;

}  // namespace

void vllm_mega_moe_fused_w4a8_up_down(
    torch::stable::Tensor const& x,         // [M, K] fp8_e4m3
    torch::stable::Tensor const& x_scale,   // [M] fp32
    torch::stable::Tensor const& w,         // [E, N, K/2] uint8 INT4 packed
    torch::stable::Tensor const& w_scale,   // [E, N, K/128] fp32
    torch::stable::Tensor const& w2,        // [E, K, (N/2)/2] uint8 INT4 packed
    torch::stable::Tensor const& w2_scale,  // [E, K, (N/2)/128] fp32
    torch::stable::Tensor const& sorted_token_ids,
    torch::stable::Tensor const& expert_ids,
    torch::stable::Tensor const& num_tokens_post_padded,
    torch::stable::Tensor const& topk_weights,  // [M, top_k] fp32
    torch::stable::Tensor& out,                 // [M, K] bf16 (zeroed)
    int64_t top_k, int64_t block_m, int64_t block_n, int64_t warp_n,
    int64_t stages, double scaling_factor) {
  STD_TORCH_CHECK(x.dim() == 2, "x must be 2D");
  STD_TORCH_CHECK(w.dim() == 3, "w must be 3D");
  STD_TORCH_CHECK(w2.dim() == 3, "w2 must be 3D");
  STD_TORCH_CHECK(out.dim() == 2, "out must be 2D");

  STD_TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  STD_TORCH_CHECK(w.is_contiguous(), "w must be contiguous");
  STD_TORCH_CHECK(w2.is_contiguous(), "w2 must be contiguous");
  STD_TORCH_CHECK(x_scale.is_contiguous(), "x_scale must be contiguous");
  STD_TORCH_CHECK(w_scale.is_contiguous(), "w_scale must be contiguous");
  STD_TORCH_CHECK(w2_scale.is_contiguous(), "w2_scale must be contiguous");
  STD_TORCH_CHECK(sorted_token_ids.is_contiguous(),
                  "sorted_token_ids must be contiguous");
  STD_TORCH_CHECK(expert_ids.is_contiguous(), "expert_ids must be contiguous");
  STD_TORCH_CHECK(topk_weights.is_contiguous(),
                  "topk_weights must be contiguous");

  STD_TORCH_CHECK(x.scalar_type() == ScalarType::Float8_e4m3fn,
                  "x must be float8_e4m3fn");
  STD_TORCH_CHECK(w.scalar_type() == ScalarType::Byte,
                  "w must be uint8 (INT4 packed)");
  STD_TORCH_CHECK(w2.scalar_type() == ScalarType::Byte,
                  "w2 must be uint8 (INT4 packed)");
  STD_TORCH_CHECK(out.scalar_type() == ScalarType::BFloat16,
                  "out must be bfloat16");

  int64_t const M = x.size(0);
  int64_t const K = x.size(1);
  int64_t const num_experts = w.size(0);
  int64_t const N = w.size(1);

  STD_TORCH_CHECK(K == w.size(2) * 2,
                  "K mismatch: x.size(1) must equal w.size(2) * 2 for "
                  "INT4-packed weights");
  STD_TORCH_CHECK(K % 128 == 0, "K must be a multiple of 128");
  STD_TORCH_CHECK(
      (block_n == 64 && warp_n == 4) || (block_n == 32 && warp_n == 8),
      "block_n/warp_n must be (64,4) or (32,8)");
  STD_TORCH_CHECK(stages > 0 && stages < 6, "stages must be in [1, 5]");
  STD_TORCH_CHECK(block_m > 0 && block_m <= 128 && block_m % 8 == 0,
                  "block_m must be in {8, 16, ..., 128}");

  torch::stable::accelerator::DeviceGuard const device_guard(
      x.get_device_index());
  cudaStream_t const stream = get_current_cuda_stream(x.get_device_index());

  fused_moe_w4a8_wgmma_up_down_acc(FusedMoeW4A8Args{
      /*x=*/reinterpret_cast<__nv_fp8_e4m3 const*>(x.data_ptr()),
      /*x_scale=*/reinterpret_cast<float const*>(x_scale.data_ptr()),
      /*w=*/reinterpret_cast<uint8_t*>(w.mutable_data_ptr()),
      /*w_scale=*/reinterpret_cast<float const*>(w_scale.data_ptr()),
      /*w2=*/reinterpret_cast<uint8_t*>(w2.mutable_data_ptr()),
      /*w2_scale=*/reinterpret_cast<float const*>(w2_scale.data_ptr()),
      /*out=*/reinterpret_cast<__nv_bfloat16*>(out.mutable_data_ptr()),
      /*sorted_token_ids=*/
      reinterpret_cast<int const*>(sorted_token_ids.data_ptr()),
      /*expert_ids=*/reinterpret_cast<int const*>(expert_ids.data_ptr()),
      /*num_tokens_post_padded=*/
      reinterpret_cast<int const*>(num_tokens_post_padded.data_ptr()),
      /*topk_weights=*/reinterpret_cast<float const*>(topk_weights.data_ptr()),
      /*top_k=*/static_cast<int>(top_k),
      /*M=*/static_cast<int>(M),
      /*K=*/static_cast<int>(K),
      /*N=*/static_cast<int>(N),
      /*num_experts=*/static_cast<int>(num_experts),
      /*sorted_num=*/static_cast<int>(sorted_token_ids.size(0)),
      /*block_m=*/static_cast<int>(block_m),
      /*block_n=*/static_cast<int>(block_n),
      /*warp_n=*/static_cast<int>(warp_n),
      /*stages=*/static_cast<int>(stages),
      /*scaling_factor=*/static_cast<float>(scaling_factor),
      /*stream=*/stream,
  });
}

STABLE_TORCH_LIBRARY_IMPL(_moe_C, CUDA, m) {
  m.impl("vllm_mega_moe_fused_w4a8_up_down",
         TORCH_BOX(&vllm_mega_moe_fused_w4a8_up_down));
}
