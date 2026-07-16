#pragma once
// SPDX-License-Identifier: Apache-2.0
// Device-side helpers shared by the vllm_mega_moe W4A8 WGMMA kernel:
// fp8 alias, swizzle/descriptor, wgmma dispatch, TMA + mbarrier, and
// the TMA-encode entry-point trampoline.

#include <cuda/std/limits>
#include <cstdint>
#include <cuda.h>
#include <cuda_fp8.h>
#include <stdio.h>
#include <cuda/ptx>
#include <cassert>
#include <cudaTypedefs.h>
#include <stdexcept>
#include <cstdio>

// CUTLASS-provided wgmma primitives for the tile sizes it covers
// (BM in {8, 16, 32, 64, 96, 128}). The remaining BM values used by
// the tuner (24, 40, 48, 56, 72, 80, 88, 104, 112, 120) are still
// hand-written below because CUTLASS does not ship those.
#include <cute/arch/mma_sm90_gmma.hpp>

#define gpuErrchk(ans)                    \
  {                                       \
    gpuAssert((ans), __FILE__, __LINE__); \
  }
#define ASSERT(cond, msg, args...) \
  assert((cond) || !fprintf(stderr, (msg "\n"), args))

inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}

// not gonna type all that
using fp8 = __nv_fp8_e4m3;

namespace aa4 {
inline PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled() {
  // Get pointer to cuTensorMapEncodeTiled
  cudaDriverEntryPointQueryResult driver_status;
  void* cuTensorMapEncodeTiled_ptr = nullptr;
  gpuErrchk(cudaGetDriverEntryPointByVersion(
      "cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, 12000,
      cudaEnableDefault, &driver_status));
  ASSERT(driver_status == cudaDriverEntryPointSuccess,
         "Failed driver status %d", 0);

  return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(
      cuTensorMapEncodeTiled_ptr);
}
}  // namespace aa4
using namespace aa4;

constexpr __device__ __forceinline__ int32_t const_ceil(float num) {
  return (static_cast<float>(static_cast<int32_t>(num)) == num)
             ? static_cast<int32_t>(num)
             : static_cast<int32_t>(num) + ((num > 0) ? 1 : 0);
}

__device__ __forceinline__ uint64_t matrix_descriptor_encode(uint64_t x) {
  return (((x) & 0x3FFFF) >> 4);
}

template <int BITS, int BASE = 4, int SHIFT = 3>
__device__ __forceinline__ int32_t swizzle(const int32_t i) {
  if constexpr (BITS == 0) return i;
  constexpr uint32_t S_MASK = ((1 << BITS) - 1) << (BASE + SHIFT);
  return i ^ ((i & S_MASK) >> SHIFT);
}

// Descriptor for a shared memory matrix.
// Implementation is derived from PTX guide:
// https://docs.nvidia.com/cuda/parallel-thread-execution/#matrix-descriptor-format
template <int SDO, int LDO, uint64_t S_MODE>
__device__ __forceinline__ uint64_t make_smem_descriptor(fp8* ptr) {
  // Convert shared memory pointer to integer
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  uint64_t desc = matrix_descriptor_encode(addr);
  desc |= ((uint64_t)LDO) << 16;
  desc |= ((uint64_t)SDO) << 32;
  // I don't think we need this anymore
  // desc |= (((uint64_t)addr >> 0x7) & 0x7) << 49;
  desc |= S_MODE << 62;
  return desc;
}

__device__ __forceinline__ float swiglu_mul(float x, float w) {
  return (x / (1 + __expf(-x))) * w;
}

__device__ __forceinline__ void warpgroup_arrive() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void warpgroup_commit_batch() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void warpgroup_wait() {
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}

template <int ScaleD, int ScaleA, int ScaleB>
__device__ __forceinline__ void wgmma24(float d[3][4], uint64_t desc_a,
                                        uint64_t desc_b) {
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n24k32.f32.e4m3.e4m3 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  "
      "%11},  "
      " %12,"
      " %13,"
      " %14, %15, %16;\n"
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
        "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
        "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3])
      : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
        "n"(int32_t(ScaleB)));
}

template <int ScaleD, int ScaleA, int ScaleB>
__device__ __forceinline__ void wgmma40(float d[5][4], uint64_t desc_a,
                                        uint64_t desc_b) {
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n40k32.f32.e4m3.e4m3 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11, "
      " %12,  %13,  %14,  %15,  %16,  %17,  %18,  %19},  "
      " %20,"
      " %21,"
      " %22, %23, %24;\n"
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
        "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
        "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
        "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3])
      : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
        "n"(int32_t(ScaleB)));
}

template <int ScaleD, int ScaleA, int ScaleB>
__device__ __forceinline__ void wgmma48(float d[6][4], uint64_t desc_a,
                                        uint64_t desc_b) {
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n48k32.f32.e4m3.e4m3 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11, "
      " %12,  %13,  %14,  %15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  "
      "%23},  "
      " %24,"
      " %25,"
      " %26, %27, %28;\n"
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
        "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
        "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
        "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
        "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3])
      : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
        "n"(int32_t(ScaleB)));
}

template <int ScaleD, int ScaleA, int ScaleB>
__device__ __forceinline__ void wgmma56(float d[7][4], uint64_t desc_a,
                                        uint64_t desc_b) {
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n56k32.f32.e4m3.e4m3 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11, "
      " %12,  %13,  %14,  %15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23, "
      " %24,  %25,  %26,  %27},  "
      " %28,"
      " %29,"
      " %30, %31, %32;\n"
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
        "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
        "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
        "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
        "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
        "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3])
      : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
        "n"(int32_t(ScaleB)));
}

template <int ScaleD, int ScaleA, int ScaleB>
__device__ __forceinline__ void wgmma72(float d[9][4], uint64_t desc_a,
                                        uint64_t desc_b) {
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11, "
      " %12,  %13,  %14,  %15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23, "
      " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  %32,  %33,  %34,  "
      "%35},  "
      " %36,"
      " %37,"
      " %38, %39, %40;\n"
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
        "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
        "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
        "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
        "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
        "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]),
        "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
        "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3])
      : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
        "n"(int32_t(ScaleB)));
}

template <int ScaleD, int ScaleA, int ScaleB>
__device__ __forceinline__ void wgmma80(float d[10][4], uint64_t desc_a,
                                        uint64_t desc_b) {
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n80k32.f32.e4m3.e4m3 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11, "
      " %12,  %13,  %14,  %15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23, "
      " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  %32,  %33,  %34,  %35, "
      " %36,  %37,  %38,  %39},  "
      " %40,"
      " %41,"
      " %42, %43, %44;\n"
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
        "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
        "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
        "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
        "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
        "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]),
        "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
        "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]),
        "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3])
      : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
        "n"(int32_t(ScaleB)));
}

template <int ScaleD, int ScaleA, int ScaleB>
__device__ __forceinline__ void wgmma88(float d[11][4], uint64_t desc_a,
                                        uint64_t desc_b) {
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n88k32.f32.e4m3.e4m3 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11, "
      " %12,  %13,  %14,  %15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23, "
      " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  %32,  %33,  %34,  %35, "
      " %36,  %37,  %38,  %39,  %40,  %41,  %42,  %43},  "
      " %44,"
      " %45,"
      " %46, %47, %48;\n"
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
        "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
        "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
        "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
        "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
        "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]),
        "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
        "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]),
        "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]),
        "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3])
      : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
        "n"(int32_t(ScaleB)));
}

template <int ScaleD, int ScaleA, int ScaleB>
__device__ __forceinline__ void wgmma104(float d[13][4], uint64_t desc_a,
                                         uint64_t desc_b) {
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n104k32.f32.e4m3.e4m3 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11, "
      " %12,  %13,  %14,  %15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23, "
      " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  %32,  %33,  %34,  %35, "
      " %36,  %37,  %38,  %39,  %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47, "
      " %48,  %49,  %50,  %51},  "
      " %52,"
      " %53,"
      " %54, %55, %56;\n"
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
        "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
        "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
        "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
        "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
        "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]),
        "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
        "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]),
        "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]),
        "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3]),
        "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]), "+f"(d[11][3]),
        "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3])
      : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
        "n"(int32_t(ScaleB)));
}

template <int ScaleD, int ScaleA, int ScaleB>
__device__ __forceinline__ void wgmma112(float d[14][4], uint64_t desc_a,
                                         uint64_t desc_b) {
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n112k32.f32.e4m3.e4m3 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11, "
      " %12,  %13,  %14,  %15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23, "
      " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  %32,  %33,  %34,  %35, "
      " %36,  %37,  %38,  %39,  %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47, "
      " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55},  "
      " %56,"
      " %57,"
      " %58, %59, %60;\n"
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
        "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
        "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
        "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
        "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
        "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]),
        "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
        "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]),
        "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]),
        "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3]),
        "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]), "+f"(d[11][3]),
        "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3]),
        "+f"(d[13][0]), "+f"(d[13][1]), "+f"(d[13][2]), "+f"(d[13][3])
      : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
        "n"(int32_t(ScaleB)));
}

template <int ScaleD, int ScaleA, int ScaleB>
__device__ __forceinline__ void wgmma120(float d[15][4], uint64_t desc_a,
                                         uint64_t desc_b) {
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n120k32.f32.e4m3.e4m3 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11, "
      " %12,  %13,  %14,  %15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23, "
      " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  %32,  %33,  %34,  %35, "
      " %36,  %37,  %38,  %39,  %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47, "
      " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  %56,  %57,  %58,  "
      "%59},  "
      " %60,"
      " %61,"
      " %62, %63, %64;\n"
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
        "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
        "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
        "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
        "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
        "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]),
        "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
        "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]),
        "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]),
        "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3]),
        "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]), "+f"(d[11][3]),
        "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3]),
        "+f"(d[13][0]), "+f"(d[13][1]), "+f"(d[13][2]), "+f"(d[13][3]),
        "+f"(d[14][0]), "+f"(d[14][1]), "+f"(d[14][2]), "+f"(d[14][3])
      : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
        "n"(int32_t(ScaleB)));
}

// wgmma dispatch by BM. For BM values that CUTLASS ships a wrapper for
// (8, 16, 32, 64, 96, 128 -- the "power-of-two-ish" set), forward to
// cute::SM90::GMMA. For the remaining tuner-friendly sizes (24, 40,
// 48, 56, 72, 80, 88, 104, 112, 120) we keep the hand-written PTX
// wrappers defined above. All call sites in the kernel pass
// <ScaleD=1, ScaleA=1, ScaleB=1>, which matches CUTLASS's SS_TN
// default (scale_D = ScaleOut::One, scaleA/scaleB = ScaleIn::One).
template <int ScaleD, int ScaleA, int ScaleB, int BM>
__device__ __forceinline__ void wgmma(float d[BM / 8][4], uint64_t desc_a,
                                      uint64_t desc_b) {
  static_assert(ScaleD == 1 && ScaleA == 1 && ScaleB == 1,
                "only <1,1,1> is currently used by the kernel");
  if constexpr (BM == 8)
    cute::SM90::GMMA::MMA_64x8x32_F32E4M3E4M3_SS_TN<>::fma(
        desc_a, desc_b, d[0][0], d[0][1], d[0][2], d[0][3]);
  else if constexpr (BM == 16)
    cute::SM90::GMMA::MMA_64x16x32_F32E4M3E4M3_SS_TN<>::fma(
        desc_a, desc_b, d[0][0], d[0][1], d[0][2], d[0][3], d[1][0], d[1][1],
        d[1][2], d[1][3]);
  else if constexpr (BM == 24)
    wgmma24<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
  else if constexpr (BM == 32)
    cute::SM90::GMMA::MMA_64x32x32_F32E4M3E4M3_SS_TN<>::fma(
        desc_a, desc_b, d[0][0], d[0][1], d[0][2], d[0][3], d[1][0], d[1][1],
        d[1][2], d[1][3], d[2][0], d[2][1], d[2][2], d[2][3], d[3][0], d[3][1],
        d[3][2], d[3][3]);
  else if constexpr (BM == 40)
    wgmma40<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
  else if constexpr (BM == 48)
    wgmma48<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
  else if constexpr (BM == 56)
    wgmma56<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
  else if constexpr (BM == 64)
    cute::SM90::GMMA::MMA_64x64x32_F32E4M3E4M3_SS_TN<>::fma(
        desc_a, desc_b, d[0][0], d[0][1], d[0][2], d[0][3], d[1][0], d[1][1],
        d[1][2], d[1][3], d[2][0], d[2][1], d[2][2], d[2][3], d[3][0], d[3][1],
        d[3][2], d[3][3], d[4][0], d[4][1], d[4][2], d[4][3], d[5][0], d[5][1],
        d[5][2], d[5][3], d[6][0], d[6][1], d[6][2], d[6][3], d[7][0], d[7][1],
        d[7][2], d[7][3]);
  else if constexpr (BM == 72)
    wgmma72<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
  else if constexpr (BM == 80)
    wgmma80<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
  else if constexpr (BM == 88)
    wgmma88<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
  else if constexpr (BM == 96)
    cute::SM90::GMMA::MMA_64x96x32_F32E4M3E4M3_SS_TN<>::fma(
        desc_a, desc_b, d[0][0], d[0][1], d[0][2], d[0][3], d[1][0], d[1][1],
        d[1][2], d[1][3], d[2][0], d[2][1], d[2][2], d[2][3], d[3][0], d[3][1],
        d[3][2], d[3][3], d[4][0], d[4][1], d[4][2], d[4][3], d[5][0], d[5][1],
        d[5][2], d[5][3], d[6][0], d[6][1], d[6][2], d[6][3], d[7][0], d[7][1],
        d[7][2], d[7][3], d[8][0], d[8][1], d[8][2], d[8][3], d[9][0], d[9][1],
        d[9][2], d[9][3], d[10][0], d[10][1], d[10][2], d[10][3], d[11][0],
        d[11][1], d[11][2], d[11][3]);
  else if constexpr (BM == 104)
    wgmma104<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
  else if constexpr (BM == 112)
    wgmma112<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
  else if constexpr (BM == 120)
    wgmma120<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
  else if constexpr (BM == 128)
    cute::SM90::GMMA::MMA_64x128x32_F32E4M3E4M3_SS_TN<>::fma(
        desc_a, desc_b, d[0][0], d[0][1], d[0][2], d[0][3], d[1][0], d[1][1],
        d[1][2], d[1][3], d[2][0], d[2][1], d[2][2], d[2][3], d[3][0], d[3][1],
        d[3][2], d[3][3], d[4][0], d[4][1], d[4][2], d[4][3], d[5][0], d[5][1],
        d[5][2], d[5][3], d[6][0], d[6][1], d[6][2], d[6][3], d[7][0], d[7][1],
        d[7][2], d[7][3], d[8][0], d[8][1], d[8][2], d[8][3], d[9][0], d[9][1],
        d[9][2], d[9][3], d[10][0], d[10][1], d[10][2], d[10][3], d[11][0],
        d[11][1], d[11][2], d[11][3], d[12][0], d[12][1], d[12][2], d[12][3],
        d[13][0], d[13][1], d[13][2], d[13][3], d[14][0], d[14][1], d[14][2],
        d[14][3], d[15][0], d[15][1], d[15][2], d[15][3]);
}

__device__ inline void load_async(fp8* dst, void const* const src_tma_map,
                                  uint64_t* bar, int global_col_idx,
                                  int global_row_idx) {
  uint64_t tma_ptr = reinterpret_cast<uint64_t>(src_tma_map);
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));

  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::"
      "bytes"
      " [%0], [%1, {%3, %4}], [%2];"
      :
      : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr), "r"(global_row_idx),
        "r"(global_col_idx)
      : "memory");
}

#define CP_ASYNC_CG(dst, src, Bytes)                                     \
  asm volatile(                                                          \
      "cp.async.cg.shared.global.L2::256B [%0], [%1], %2;\n" ::"r"(dst), \
      "l"(src), "n"(Bytes))

#define CP_ASYNC_CG4(dst, src, Bytes)                                   \
  asm volatile(                                                         \
      "cp.async.ca.shared.global.L2::64B [%0], [%1], %2;\n" ::"r"(dst), \
      "l"(src), "n"(Bytes))

__device__ __forceinline__ void init_barrier(uint64_t* bar, int thread_count,
                                             int transaction_count) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar_ptr),
               "r"(thread_count + transaction_count));
}

__device__ __forceinline__ void cp_async_mbarrier_arrive(uint64_t* bar) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" ::"r"(bar_ptr));
}

__device__ __forceinline__ void arrive(uint64_t* bar, uint32_t count = 1) {
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.arrive.shared.b64 _, [%0], %1;\n"
               :
               : "r"(mbar_ptr), "r"(count)
               : "memory");
}

__device__ __forceinline__ void expect_bytes(uint64_t* bar, uint32_t bytes) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n" ::
          "r"(bar_ptr),
      "r"(bytes));
}

__device__ __forceinline__ void wait(uint64_t* bar, int kPhaseBit) {
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "{\n"
      ".reg .pred                P1;\n"
      "WAIT:\n"
      "mbarrier.try_wait.parity.shared.b64 P1, [%0], %1;\n"
      "@!P1                       bra.uni WAIT;\n"
      "}\n" ::"r"(mbar_ptr),
      "r"(kPhaseBit));
}

__device__ __forceinline__ void st_matrix_x4_trans(uint32_t* tile,
                                                   uint32_t mat) {
  asm volatile(
      "stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};"
      :
      : "r"(mat), "r"(tile[0]), "r"(tile[1]), "r"(tile[2]), "r"(tile[3])
      : "memory");
}
