// ============================================================================
// Fused MoE w4a8 (INT4 weight + FP8 activation) WGMMA up/down kernel.
//
// UP and DOWN projections both consume INT4-packed weights that are
// dequantized to FP8 in shared memory (lop3 + prmt LUT) before WGMMA.
// Common device-side helpers (fp8 alias, swizzle/descriptor, wgmma, TMA,
// barriers) live in vllm_mega_moe_common.cuh; per-kernel SMEM layouts
// (smem_up_w4a8, smem_down_w4a8) are defined here.
#include "vllm_mega_moe_common.cuh"
// ============================================================================

// ============================================================================
// INT4 → FP8 dequantization helpers (lop3 + prmt)
// ============================================================================

__device__ __forceinline__ void make_fp8_lut(
    float scale,
    uint32_t& neg_lo, uint32_t& neg_hi,
    uint32_t& pos_lo, uint32_t& pos_hi)
{
    uint8_t neg[8], pos[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        fp8 nv = fp8(float(i - 8) * scale);
        fp8 pv = fp8(float(i) * scale);
        neg[i] = *reinterpret_cast<uint8_t*>(&nv);
        pos[i] = *reinterpret_cast<uint8_t*>(&pv);
    }
    neg_lo = *reinterpret_cast<uint32_t*>(neg);
    neg_hi = *reinterpret_cast<uint32_t*>(neg + 4);
    pos_lo = *reinterpret_cast<uint32_t*>(pos);
    pos_hi = *reinterpret_cast<uint32_t*>(pos + 4);
}

__device__ __forceinline__ void dequant_8xint4_to_8xfp8(
    uint32_t packed,
    uint32_t neg_lo, uint32_t neg_hi,
    uint32_t pos_lo, uint32_t pos_hi,
    uint32_t& out_lo, uint32_t& out_hi)
{
    static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
    uint32_t sign;
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(sign)
        : "r"(packed), "n"(0x88888888), "n"(0x64206420), "n"(immLut));
    sign >>= 1;

    uint32_t lut_idx = packed & 0x77777777;

    asm volatile(
        "{\n"
        "  .reg .b32 pos, neg;\n"
        "  prmt.b32 neg, %3, %4, %1;\n"
        "  prmt.b32 pos, %5, %6, %1;\n"
        "  prmt.b32 %0, pos, neg, %2;\n"
        "}\n"
        : "=r"(out_lo)
        : "r"(lut_idx), "r"(sign),
          "r"(neg_lo), "r"(neg_hi), "r"(pos_lo), "r"(pos_hi));

    asm volatile(
        "{\n"
        "  .reg .b32 pos, neg;\n"
        "  prmt.b32 neg, %3, %4, %1;\n"
        "  prmt.b32 pos, %5, %6, %1;\n"
        "  prmt.b32 %0, pos, neg, %2;\n"
        "}\n"
        : "=r"(out_hi)
        : "r"(lut_idx >> 16), "r"(sign >> 16),
          "r"(neg_lo), "r"(neg_hi), "r"(pos_lo), "r"(pos_hi));
}

// ----- w4a8 SMEM layouts -----
// ============================================================================
// w4a8 SMEM layout: packed INT4 + FP8 double-buffered + activations
// ============================================================================
template<int STAGES, int WN, int BM, int BK, int BN>
struct smem_up_w4a8
{
    // Packed INT4 weights: [STAGES * WN * BN * BK/2]
    // TMA cp.async.bulk.tensor requires 16B alignment; use 128B for swizzle modes.
    alignas(128) uint8_t w_packed[STAGES*WN*BN*(BK/2)];
    // Dequantized FP8 weights (double-buffered): [2 * WN * BK * BN]
    // wgmma SMEM descriptor requires 128B alignment (swizzle base).
    alignas(128) fp8 w_fp8[2*WN*BK*BN];
    // FP8 activations: [STAGES * BK * BM]
    alignas(128) fp8 x[STAGES*BK*BM];
    // Scales
    // ----- per-row + per-token paradigm (W4A8_PER_ROW_PER_TOKEN migration) -----
    // weight scale: per-row, K-group-128 -> WN*BN rows * STAGES K-groups, ping-pong 2x.
    // Layout (Iteration 2): [ping_pong][row_in_block][stage_in_window]  (row-major along STAGES dim,
    // so producer threads with consecutive threadIdx.x read consecutive 4B global addresses ->
    // coalesced LDGSTS; see optimization.md Iteration 2).
    float scale_w_up[2*STAGES * WN*BN];
    // activation scale: per-token, shared across all K -> just BM scalars
    // (no STAGES dim: a single scale per token covers the entire K reduction)
    float scale_x_up[BM];
    // Bridge for passing token_scale from up consumer to down consumer
    float token_scale_bridge[BM];
};

// w4a8 DOWN SMEM layout: packed INT4 + FP8 double-buffered + x + out + bridge
template<int STAGES, int WN, int BM, int BK, int BN>
struct smem_down_w4a8
{
    // DOWN dimensions: BK2 = WN*BN/2, BN2 = BK*2
    // Packed INT4 weights: [STAGES * BN2 * (BK2/2)]
    alignas(128) uint8_t w_packed[STAGES * (BK*2) * (WN*BN/4)];
    // Dequantized FP8 weights (double-buffered): [2 * BN2 * BK2]
    alignas(128) fp8 w_fp8[2 * (BK*2) * (WN*BN/2)];
    // x (SwiGLU output, written by UP consumer): [BM * K2] = [BM * WN*BN/2]
    alignas(128) fp8 x[BM * WN * BN / 2];
    // Output buffer with padding: [BM * (BN2+8)] = [BM * (BK*2+8)]
    alignas(16) __nv_bfloat16 out[BM * (BK*2 + 8)];
    // Bridge for passing token_scale from UP consumer to DOWN consumer
    float token_scale_bridge[BM];
};



// ----- w4a8 kernel + launch/dispatch wrappers -----
// ============================================================================
// w4a8 kernel: up projection with INT4 packed weights, down projection FP8
// ============================================================================
// Launch bounds: 2nd param picks min_blocks_per_sm to trade reg/occupancy.
// STAGES=1 + WN=4 has small SMEM (~98KB) and small fragment register footprint
// (TN=BN*WN/(64*WARPGROUPS) is small), so 2 blocks/SM fit and helps. WN=8 needs
// too many fragment regs even at STAGES=1; force 1 block/SM.
template <int BM, int BK, int BN, int WN, int STAGES, int PRODUCER_THREADS>
__global__ __launch_bounds__(WN*32 + PRODUCER_THREADS,
                             (STAGES==1 && WN<=4) ? 2 : 1) void fused_moe_w4a8_wgmma_up_down_acc_kernel(
        const fp8* __restrict__ x,
        const float* __restrict__ x_scale,
        const __grid_constant__ CUtensorMap tensor_map_w_packed,  // INT4 packed [E*N, K/2]
        const float* __restrict__ w_scale,
        const __grid_constant__ CUtensorMap tensor_map_w2_packed,  // INT4 packed [E*N2, K2/2]
        const float* __restrict__ w2_scale,
        __nv_bfloat16* __restrict__ out,
        const int* __restrict__ sorted_token_ids,
        const int* __restrict__ expert_ids,
        const int* __restrict__ num_tokens_post_padded,
        const float* __restrict__ topk_weights,
        const int top_k,
        int M,
        int K,
        int N,
        float scaling_factor
        )
{
    constexpr int CONSUMER_THREADS = WN*32;
    constexpr int WARPGROUPS = WN / 4;
    const int32_t warpM = blockIdx.y;
    const int exp_idx = expert_ids[warpM];
    if(warpM * BM >= num_tokens_post_padded[0])
        return;

    constexpr int block_shape[2] = {128, 128};

    // Up projection uses 128B swizzle for FP8 data
    constexpr uint32_t S_BITS_UP = 3;
    constexpr uint32_t S_MODE_UP = 4 - S_BITS_UP;

    constexpr uint32_t S_BITS_DOWN = WN*BN == 256 ? 3 :
                                     WN*BN == 128 ? 2 :
                                                    1;
    constexpr uint32_t S_MODE_DOWN = 4 - S_BITS_DOWN;

    const int K2 = N/2;
    const int N2 = K;
    const int lane_id = threadIdx.x%32;
    const bool is_producer = threadIdx.x < PRODUCER_THREADS;
    const int warp_id = is_producer ? threadIdx.x/32 : (threadIdx.x-PRODUCER_THREADS)/32;

    constexpr int BK2 = WN*BN/2;
    constexpr int BN2 = BK*2;

    // SMEM sizes (UP)
    constexpr int WS = WN*BK*BN;            // FP8 weight stage size (UP)
    constexpr int PACKED_WS = WN*BN*(BK/2); // INT4 packed weight stage size (UP)
    constexpr int XS = BK*BM;
    constexpr int TB = 16;
    constexpr int TO = TB/sizeof(fp8);
    constexpr int TPT = BM/4;

    // DOWN sizes
    constexpr int WS_DOWN = BN2 * BK2;                 // FP8 weight stage size (DOWN)
    constexpr int PACKED_WS_DOWN = BN2 * (BK2/2);      // INT4 packed weight stage size (DOWN)
    constexpr int TOTAL_UINT32_DOWN = PACKED_WS_DOWN / 4;

    constexpr int TN2 = BN2/(64*WARPGROUPS);
    constexpr int scales_per_stage_down = BN2 / block_shape[0];

    extern __shared__ __align__(1024) uint8_t sh[];
    smem_up_w4a8<STAGES, WN, BM, BK, BN>& s = *reinterpret_cast<smem_up_w4a8<STAGES, WN, BM, BK, BN>*>(sh);

    // 4 groups of barriers (UP and DOWN have separate barrier resources):
    //   bar[0..S-1]   = UP bar_load (TMA + cp.async done, data ready for consumer)
    //   bar[S..2S-1]  = UP bar_release (consumer done, producer can reuse stage)
    //   bar[2S..3S-1] = DOWN bar_load (TMA + cp.async done, data ready for consumer)
    //   bar[3S..4S-1] = DOWN bar_release (consumer done, producer can reuse stage)
    // Separate barrier resources allow producer to start DOWN TMA loads immediately
    // after UP loop finishes, overlapping with consumer's SwiGLU + FP8 quantization.
    constexpr int S = STAGES;
    __shared__ __align__(8) uint64_t bar[4*S];
    __shared__ float topk_scales[BM];
    // per-row scale: 2 (ping-pong) * STAGES windows * BN2 rows per stage.
    // BN2 = BK*2. Each row carries a single fp32 K2-group-128 scale (the
    // group is selected by blockIdx.x, see producer DOWN-stage load).
    // NOTE: current implementation requires BK2 (= WN*BN/2) == 128 so that
    // exactly one K2-group lies inside the consumer's per-stage reduction.
    // Layout (Iteration 2): [ping_pong][row_in_block][stage_in_window]  (innermost dim
    // is STAGES so producer LDGSTS along K-groups axis is coalesced).
    __shared__ float scale_w_down[2*STAGES * (BK*2)];

    // Alias for DOWN barriers (bar_down[0..S-1] = load, bar_down[S..2S-1] = release)
    uint64_t* bar_down = bar + 2*S;

    if (threadIdx.x == 0)
    {
        for (int i = 0; i < S; i++)
        {
            // UP barriers
            init_barrier(&bar[i], PRODUCER_THREADS + 1, 0);       // UP bar_load
            init_barrier(&bar[S + i], CONSUMER_THREADS, 0);        // UP bar_release
            // DOWN barriers
            init_barrier(&bar_down[i], PRODUCER_THREADS + 1, 0);  // DOWN bar_load
            init_barrier(&bar_down[S + i], CONSUMER_THREADS, 0);   // DOWN bar_release
        }
    }
    __syncthreads();

    // Full consumer sync (all CONSUMER_THREADS) — needed for cross-warpgroup data sharing
    // (e.g., SwiGLU block_max reduction, token_scale bridge).
    auto consumer_sync = [&]()
    {
        asm volatile("bar.sync 1, %0;\n" :: "n"(CONSUMER_THREADS));
    };

    // Per-warpgroup dequant sync — only syncs within a single warpgroup (128 threads).
    // Safe because stride-1 dequant mapping partitions w_fp8 writes by warpgroup:
    //   warpgroup 0 (cons_tid 0..127) writes fp8 bytes 0..WS/2-1
    //   warpgroup 1 (cons_tid 128..255) writes fp8 bytes WS/2..WS-1
    // Each warpgroup's WGMMA only reads from its own write region.
    auto dequant_sync = [&]()
    {
        if constexpr (WARPGROUPS >= 2) {
            const int wg_id = (threadIdx.x - PRODUCER_THREADS) / 128;
            if (wg_id == 0)
                asm volatile("bar.sync 3, %0;\n" :: "n"(CONSUMER_THREADS / WARPGROUPS));
            else
                asm volatile("bar.sync 4, %0;\n" :: "n"(CONSUMER_THREADS / WARPGROUPS));
        } else {
            asm volatile("bar.sync 1, %0;\n" :: "n"(CONSUMER_THREADS));
        }
    };

    int n_stages_up = K/block_shape[0];
    int n_stages_down = N2/BN2;

    constexpr int TM = BM/8;
    constexpr int TN = BN/16;
    nv_bfloat162 f_acc[TN][TM][2];
    memset(f_acc, 0, sizeof(f_acc));

    // Dequant constants (used by consumer-side dequant)
    constexpr int TOTAL_UINT32 = PACKED_WS / 4;  // total uint32 words per packed stage

    // ========== UP PROJECTION ==========
    if (is_producer)
    {
        constexpr int X_IT = const_ceil(XS/float(PRODUCER_THREADS*TO));
        int tsrc[X_IT];
        int i = (threadIdx.x)*TO;
        for(int r = 0; r < X_IT; r += 1)
        {
            if (i < XS)
            {
                int tdest = sorted_token_ids[warpM*BM + r*(PRODUCER_THREADS/8) + threadIdx.x/8];
                tsrc[r] = tdest / top_k;
                if(threadIdx.x % 8 == 0 && tsrc[r] < M && i < XS)
                {
                    uint32_t smem = __cvta_generic_to_shared(&topk_scales[r*(PRODUCER_THREADS/8) + threadIdx.x/8]);
                    CP_ASYNC_CG4(smem, topk_weights + tdest, 4);
                }
            }
            i += PRODUCER_THREADS*TO;
        }

        // Unified loop (2-barrier producer/consumer pipeline):
        // Consumer initial-arrives bar_release, so first wait passes immediately.
        // Producer: wait bar_release → cp.async + TMA → signal bar_load.
        int smem_stage = 0;
        const int w_row_up = exp_idx * N + (blockIdx.x)*WN*BN;
        int p = 0;
        for (int kt = 0; kt < n_stages_up; kt++)
        {
            if (smem_stage == STAGES)
            {
                p ^= 1;
                smem_stage = 0;
            }

            // Wait for consumer to release this stage (first time: instant because consumer init-arrived)
            wait(bar + S + smem_stage, p);

            int k_off = kt * block_shape[0];
            int ii = (threadIdx.x)*TO;
            int col = k_off + ii%BK;
            int swizzled_x = swizzle<S_BITS_UP>(ii);

            for(int r = 0; r < X_IT; r += 1)
            {
                int row = tsrc[r];
                if(row < M && ii < XS)
                {
                    uint32_t sm = __cvta_generic_to_shared(s.x + smem_stage*XS + swizzled_x);
                    CP_ASYNC_CG(sm, reinterpret_cast<const float4*>(x + row*K + col), TB);
                    // ----- per-token activation scale (new paradigm) -----
                    // Load only once across the entire UP K reduction: only at kt==0
                    // (first pass through stages). One scale per token; layout is
                    // s.scale_x_up[row_in_block]. Each row is loaded by exactly one
                    // producer thread (threadIdx.x % 8 == 0) on this row's iteration.
                    if(kt == 0 && smem_stage == 0 && threadIdx.x % 8 == 0)
                    {
                        int row_in_block = r * (PRODUCER_THREADS/8) + threadIdx.x/8;
                        if (row_in_block < BM)
                        {
                            uint32_t smem_sc = __cvta_generic_to_shared(s.scale_x_up + row_in_block);
                            CP_ASYNC_CG4(smem_sc, &x_scale[row], 4);
                        }
                    }
                }
                ii += PRODUCER_THREADS*TO;
                swizzled_x += PRODUCER_THREADS*TO;
            }

            // ----- per-row weight scale (new paradigm, Iteration 2 coalesced layout) -----
            // w_scale logical shape: [E, N, K_groups]  (N already interleaved gate/up rep=8 on Python side)
            // SMEM layout: s.scale_w_up[ping_pong][row_in_block][stage_in_window]
            //   size = 2 * STAGES * (WN*BN)
            //   addr = ((kt/STAGES)%2)*STAGES*(WN*BN) + row_in_block*STAGES + stage_in_window
            // Thread mapping: idx = it*PRODUCER_THREADS + threadIdx.x
            //   stage_in_window = idx % STAGES   (consecutive threads consume consecutive K_groups,
            //                                     i.e. consecutive 4B in global memory along the
            //                                     K_groups axis => coalesced 32B sectors)
            //   row_in_block    = idx / STAGES
            // n_start_row = blockIdx.x * (WN*BN)  (matches w_row_up offset above)
            const int K_groups = K / block_shape[0];     // K / 128
            const int n_start = blockIdx.x * (WN*BN);
            if (smem_stage == 0)
            {
                constexpr int SW_TOTAL = STAGES * (WN*BN);  // total scales to load this window
                #pragma unroll
                for (int it = 0; it * PRODUCER_THREADS < SW_TOTAL; it++)
                {
                    int idx = it * PRODUCER_THREADS + threadIdx.x;
                    if (idx < SW_TOTAL)
                    {
                        int stage_in_window = idx % STAGES;
                        int row_in_block    = idx / STAGES;
                        if (kt + stage_in_window < n_stages_up)
                        {
                            uint32_t smem_sc = __cvta_generic_to_shared(
                                s.scale_w_up + ((kt/STAGES)%2) * SW_TOTAL
                                             + row_in_block * STAGES
                                             + stage_in_window);
                            CP_ASYNC_CG4(smem_sc,
                                &w_scale[exp_idx * N * K_groups
                                       + (n_start + row_in_block) * K_groups
                                       + (kt + stage_in_window)], 4);
                        }
                    }
                }
            }
            cp_async_mbarrier_arrive(bar + smem_stage);

            // TMA load packed weights (INT4)
            if(threadIdx.x == 0)
            {
                expect_bytes(bar + smem_stage, PACKED_WS*sizeof(uint8_t));
                load_async(reinterpret_cast<fp8*>(s.w_packed + smem_stage*PACKED_WS),
                           &tensor_map_w_packed, bar + smem_stage, w_row_up, k_off/2);
            }
            smem_stage++;
        }

        // ========== SEAMLESS TRANSITION: UP → DOWN ==========
        // No __syncthreads() needed! DOWN barriers are already initialized.
        // Producer starts DOWN TMA loads immediately; consumer may still be doing
        // SwiGLU + FP8 quantization. This is safe because:
        //   - Producer writes to s_d.w_packed (SMEM offset 0..PACKED_WS_DOWN*STAGES-1)
        //   - Consumer writes to s_d.x (higher SMEM offset) and s_d.out (even higher)
        //   - These SMEM regions do NOT overlap
        // The bar_down[S+i] (DOWN bar_release) initial-arrive from consumer
        // gates when producer can start (consumer arrives before or during SwiGLU).
        {
            smem_down_w4a8<STAGES, WN, BM, BK, BN>& s_d = *reinterpret_cast<smem_down_w4a8<STAGES, WN, BM, BK, BN>*>(sh);

            // per-row paradigm: w2_scale shape is [E, N2, K2/128].
            //   N2 axis = output K (per-row),   K2 axis (intermediate I) per-group-128.
            // We pick K2-group = blockIdx.x (assuming BK2 == 128 -> exactly one group
            // per consumer reduction). Each DOWN stage loads BN2 rows from N2,
            // starting at kt*BN2.
            const int K2_groups = K2 / block_shape[0];

            int smem_stage_down = 0;
            int p_down = 0;
            for (int kt = 0; kt < n_stages_down; kt++)
            {
                if (smem_stage_down == STAGES)
                {
                    p_down ^= 1;
                    smem_stage_down = 0;
                }

                // Wait for consumer to release this stage (first time: instant after initial arrive)
                wait(bar_down + S + smem_stage_down, p_down);

                // Load w2_scale via cp.async  (per-row, group-128 along K2)
                // Each row carries one fp32 scale at K2-group = blockIdx.x.
                // SMEM layout (Iteration 2): [ping_pong][row_in_block][stage_in_window]
                //   addr = ((kt/STAGES)%2)*STAGES*BN2 + row_in_block*STAGES + stage_in_window
                // Window load: at smem_stage_down==0, fetch STAGES stages worth
                // of (BN2 rows) at once. Thread mapping puts stage_in_window in the
                // innermost dim so consecutive threads consume consecutive K-groups
                // in global memory => coalesced 32B sectors.
                if (smem_stage_down == 0)
                {
                    constexpr int SW_TOTAL_DOWN = STAGES * BN2;
                    #pragma unroll
                    for (int it = 0; it * PRODUCER_THREADS < SW_TOTAL_DOWN; it++)
                    {
                        int idx = it * PRODUCER_THREADS + threadIdx.x;
                        if (idx < SW_TOTAL_DOWN)
                        {
                            int stage_in_window = idx % STAGES;
                            int row_in_block    = idx / STAGES;
                            if (kt + stage_in_window < n_stages_down)
                            {
                                uint32_t smem_sc = __cvta_generic_to_shared(
                                    scale_w_down + ((kt/STAGES)%2) * SW_TOTAL_DOWN
                                                 + row_in_block * STAGES
                                                 + stage_in_window);
                                CP_ASYNC_CG4(smem_sc,
                                    &w2_scale[exp_idx * N2 * K2_groups
                                            + ((kt + stage_in_window) * BN2 + row_in_block) * K2_groups
                                            + blockIdx.x * (BK2/block_shape[0])], 4);
                            }
                        }
                    }
                }
                cp_async_mbarrier_arrive(bar_down + smem_stage_down);

                // TMA load packed INT4 weights
                if(threadIdx.x == 0)
                {
                    const int w_row_down = blockIdx.x*BK2/2;
                    const int w_col_down = exp_idx * N2 + kt*BN2;
                    expect_bytes(bar_down + smem_stage_down, PACKED_WS_DOWN*sizeof(uint8_t));
                    load_async(reinterpret_cast<fp8*>(s_d.w_packed + smem_stage_down*PACKED_WS_DOWN),
                               &tensor_map_w2_packed, bar_down + smem_stage_down, w_col_down, w_row_down);
                }
                smem_stage_down++;
            }
        }
    }
    // CONSUMER UP: when STAGES>=2, pipeline dequant(N+1) with WGMMA(N);
    //              when STAGES==1, use original sequential dequant→WGMMA.
    else
    {
        int token_src = M;
        if(threadIdx.x < PRODUCER_THREADS + BM)
            token_src = sorted_token_ids[warpM*BM + threadIdx.x-PRODUCER_THREADS] / top_k;

        // Consumer-side dequant constants
        constexpr int CONSUMER_DQ_PER_THREAD = TOTAL_UINT32 / CONSUMER_THREADS;
        const int cons_tid = threadIdx.x - PRODUCER_THREADS;  // consumer-local thread id

        // Precompute FP8 LUT (identity scale, actual w_scale applied after WGMMA)
        uint32_t neg_lo, neg_hi, pos_lo, pos_hi;
        make_fp8_lut(1.0f, neg_lo, neg_hi, pos_lo, pos_hi);

        // Release initial bar_release so producer can start loading
        for (int i = 0; i < S; i++)
            arrive(&bar[S + i]);

        int p_load = 0;

        // Lambda: dequant a given stage into w_fp8[dq_slot]
        // uint4 (16B) merged writes: each thread processes PAIRS of consecutive widx values
        // (even, even+1). Since swizzle guarantees even-widx pairs are always adjacent and
        // 16B-aligned after swizzle, we can merge two uint2 (8B) writes into one uint4 (16B)
        // write, halving the store instruction count (8 vs 16 per thread).
        // Stride-1 pair mapping preserves bank-conflict optimization.
        auto do_dequant_up = [&](int smem_slot, int dq_slot) __attribute__((always_inline)) {
            uint32_t* src = reinterpret_cast<uint32_t*>(s.w_packed + smem_slot * PACKED_WS);
            constexpr int WARP_SIZE = 32;
            constexpr int PAIRS_PER_THREAD = CONSUMER_DQ_PER_THREAD / 2;
            const int warp_in_cons = cons_tid / WARP_SIZE;
            const int lane_in_warp = cons_tid % WARP_SIZE;
            const int pair_base = warp_in_cons * (WARP_SIZE * PAIRS_PER_THREAD) + lane_in_warp;
            // Process 4 pairs per iteration for maximum ILP (2 iterations total)
            #pragma unroll
            for (int r = 0; r < PAIRS_PER_THREAD; r += 4)
            {
                int pidx0 = pair_base + r * WARP_SIZE;
                int pidx1 = pidx0 + WARP_SIZE;
                int pidx2 = pidx0 + 2 * WARP_SIZE;
                int pidx3 = pidx0 + 3 * WARP_SIZE;
                // Each pair reads 2 consecutive uint32 (even, odd)
                uint32_t packed0e = src[pidx0 * 2];
                uint32_t packed0o = src[pidx0 * 2 + 1];
                uint32_t packed1e = src[pidx1 * 2];
                uint32_t packed1o = src[pidx1 * 2 + 1];
                uint32_t packed2e = src[pidx2 * 2];
                uint32_t packed2o = src[pidx2 * 2 + 1];
                uint32_t packed3e = src[pidx3 * 2];
                uint32_t packed3o = src[pidx3 * 2 + 1];
                // Dequant all 8 (ILP: all independent)
                uint32_t lo0e, hi0e, lo0o, hi0o;
                uint32_t lo1e, hi1e, lo1o, hi1o;
                uint32_t lo2e, hi2e, lo2o, hi2o;
                uint32_t lo3e, hi3e, lo3o, hi3o;
                dequant_8xint4_to_8xfp8(packed0e, neg_lo, neg_hi, pos_lo, pos_hi, lo0e, hi0e);
                dequant_8xint4_to_8xfp8(packed0o, neg_lo, neg_hi, pos_lo, pos_hi, lo0o, hi0o);
                dequant_8xint4_to_8xfp8(packed1e, neg_lo, neg_hi, pos_lo, pos_hi, lo1e, hi1e);
                dequant_8xint4_to_8xfp8(packed1o, neg_lo, neg_hi, pos_lo, pos_hi, lo1o, hi1o);
                dequant_8xint4_to_8xfp8(packed2e, neg_lo, neg_hi, pos_lo, pos_hi, lo2e, hi2e);
                dequant_8xint4_to_8xfp8(packed2o, neg_lo, neg_hi, pos_lo, pos_hi, lo2o, hi2o);
                dequant_8xint4_to_8xfp8(packed3e, neg_lo, neg_hi, pos_lo, pos_hi, lo3e, hi3e);
                dequant_8xint4_to_8xfp8(packed3o, neg_lo, neg_hi, pos_lo, pos_hi, lo3o, hi3o);
                // Write 4 merged uint4 (16B each) — even-widx swizzle addr is always 16B-aligned
                int fp8_off = dq_slot * WS;
                int sw0 = swizzle<S_BITS_UP>(fp8_off + (pidx0 * 2) * 8);
                int sw1 = swizzle<S_BITS_UP>(fp8_off + (pidx1 * 2) * 8);
                int sw2 = swizzle<S_BITS_UP>(fp8_off + (pidx2 * 2) * 8);
                int sw3 = swizzle<S_BITS_UP>(fp8_off + (pidx3 * 2) * 8);
                *reinterpret_cast<uint4*>(s.w_fp8 + sw0) = make_uint4(lo0e, hi0e, lo0o, hi0o);
                *reinterpret_cast<uint4*>(s.w_fp8 + sw1) = make_uint4(lo1e, hi1e, lo1o, hi1o);
                *reinterpret_cast<uint4*>(s.w_fp8 + sw2) = make_uint4(lo2e, hi2e, lo2o, hi2o);
                *reinterpret_cast<uint4*>(s.w_fp8 + sw3) = make_uint4(lo3e, hi3e, lo3o, hi3o);
            }
        };

        if constexpr (STAGES >= 2)
        {
            // ---- Pipelined mode: prologue dequant stage 0, then overlap dequant(N+1) with WGMMA(N) ----
            {
                wait(bar + 0, p_load);
                do_dequant_up(0, 0);
                // No dequant_sync needed: wgmma.fence.sync.aligned (warpgroup_arrive)
                // at the start of the first loop iteration provides warpgroup-wide
                // synchronization and guarantees all dequant stores are visible to WGMMA.
            }

            for (int compute_stage = 0; compute_stage < n_stages_up; compute_stage += 1)
            {
                int ss = compute_stage % STAGES;
                int dq_slot = compute_stage % 2;

                // ----- per-token scale_x (new paradigm) -----
                // scale_x_up[BM] is loaded once and shared across all K-groups.
                // Index by lane mapping (same as before): token_idx = (t/2)*8 + (lane_id%4)*2
                float scale_x[TPT];
                for(int t = 0; t < TPT; t+=2)
                {
                    int token_idx = (t/2)*8 + (lane_id%4)*2;
                    float2 sx = *reinterpret_cast<const float2*>(&s.scale_x_up[token_idx]);
                    scale_x[t] = sx.x;
                    scale_x[t+1] = sx.y;
                }
                // ----- per-row weight scale (new paradigm, Iteration 2 layout) -----
                // scale_w_up SMEM layout: [ping_pong][row_in_block][stage_in_window]
                //   addr = ((compute_stage/STAGES)%2) * STAGES*(WN*BN)
                //        + n_row_in_block * STAGES + (compute_stage%STAGES)
                // Per wgmma fragment lane mapping (PTX m16n8k16 D layout, see plan doc):
                //   tile_acc[tn][tm][0,1] -> weight N-row = warpgroup_start + tn*64 + (warp%4)*16 + (lane/4)
                //   tile_acc[tn][tm][2,3] -> same + 8 (the rep=8 interleaved up row vs gate row)
                // warpgroup_start = (warp_id/4) * 4 * BN  (each warpgroup spans 4*BN rows of weight N)
                float scale_w[TN][2];
                {
                    int ping_pong = ((compute_stage/STAGES)%2) * STAGES * (WN*BN);
                    int stage_in_window = compute_stage % STAGES;
                    int wg_start = (warp_id/4) * 4 * BN;
                    int warp_off = (warp_id%4) * 16 + (lane_id/4);
                    for (int tn = 0; tn < TN; tn++) {
                        int n_row_gate = wg_start + tn*64 + warp_off;
                        scale_w[tn][0] = s.scale_w_up[ping_pong + n_row_gate * STAGES + stage_in_window];
                        scale_w[tn][1] = s.scale_w_up[ping_pong + (n_row_gate + 8) * STAGES + stage_in_window];
                    }
                }

                float tile_acc[TN][TM][4];
                memset(tile_acc, 0, sizeof(tile_acc));
                for (int tn = 0; tn < TN; tn++)
                    for (int tm = 0; tm < TM; tm++)
                        for (int t = 0; t < 4; t++)
                            asm volatile("" : "+f"(tile_acc[tn][tm][t]) :: "memory");

                // Launch WGMMA (reads w_fp8[dq_slot] + x[ss])
                cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
                warpgroup_arrive();
                for(int tn = 0; tn<TN; tn++)
                {
                    fp8* sw_ptr = s.w_fp8 + dq_slot*WS + (warp_id/4)*(BN*4)*BK + tn*64*BK;
                    fp8* sx_ptr = s.x + ss*XS;
                    uint64_t desc_w = make_smem_descriptor<64, 1, S_MODE_UP>(sw_ptr);
                    uint64_t desc_x = make_smem_descriptor<64, 1, S_MODE_UP>(sx_ptr);
                    wgmma<1,1,1, BM>(tile_acc[tn], desc_w, desc_x);
                    wgmma<1,1,1, BM>(tile_acc[tn], desc_w+1*(32>>4), desc_x+1*(32>>4));
                    wgmma<1,1,1, BM>(tile_acc[tn], desc_w+2*(32>>4), desc_x+2*(32>>4));
                    wgmma<1,1,1, BM>(tile_acc[tn], desc_w+3*(32>>4), desc_x+3*(32>>4));
                }
                warpgroup_commit_batch();

                // Overlap: dequant next stage while WGMMA is in-flight.
                // Safe because WGMMA reads w_fp8[dq_slot] and x[ss], while dequant writes
                // w_fp8[next_dq_slot] (double-buffered) and reads w_packed[next_ss] (different slot).
                // We must NOT arrive bar_release[ss] yet (x[ss] still read by WGMMA).
                if (compute_stage + 1 < n_stages_up)
                {
                    int next_ss = (compute_stage + 1) % STAGES;
                    int next_dq_slot = (compute_stage + 1) % 2;

                    // Compute next phase parity for bar_load
                    int next_p_load = p_load;
                    if (next_ss == 0)
                        next_p_load = p_load ^ 1;

                    wait(bar + next_ss, next_p_load);
                    do_dequant_up(next_ss, next_dq_slot);
                    // No dequant_sync: wgmma.fence.sync.aligned in next iteration handles it
                }

                warpgroup_wait();

                // Release stage for producer (bar_release)
                arrive(bar + S + ss);

                // Scale and accumulate (per-row weight scale: indexed by tn)
                for(int tm = 0; tm<TM; tm++)
                {
                    for(int tn = 0; tn<TN; tn++)
                    {
                        f_acc[tn][tm][0] = __hadd2(f_acc[tn][tm][0],
                                __nv_bfloat162(
                                    scale_w[tn][0] * scale_x[tm*2] * tile_acc[tn][tm][0],
                                    scale_w[tn][0] * scale_x[tm*2 + 1] * tile_acc[tn][tm][1]
                                    )
                                );
                        f_acc[tn][tm][1] = __hadd2(f_acc[tn][tm][1],
                                __nv_bfloat162(
                                    scale_w[tn][1] * scale_x[tm*2] * tile_acc[tn][tm][2],
                                    scale_w[tn][1] * scale_x[tm*2 + 1] * tile_acc[tn][tm][3]
                                    )
                                );
                    }
                }

                if (ss == STAGES - 1)
                    p_load ^= 1;
            }
        }
        else  // STAGES == 1: sequential dequant→WGMMA (no overlap possible)
        {
            for (int compute_stage = 0; compute_stage < n_stages_up; compute_stage += 1)
            {
                int ss = compute_stage % STAGES;
                int dq_slot = compute_stage % 2;

                wait(bar + ss, p_load);
                do_dequant_up(ss, dq_slot);
                // No dequant_sync: wgmma.fence.sync.aligned below handles it

                float scale_x[TPT];
                for(int t = 0; t < TPT; t+=2)
                {
                    int token_idx = (t/2)*8 + (lane_id%4)*2;
                    // per-token scale (new paradigm): no STAGES dim
                    float2 sx = *reinterpret_cast<const float2*>(&s.scale_x_up[token_idx]);
                    scale_x[t] = sx.x;
                    scale_x[t+1] = sx.y;
                }
                float scale_w[TN][2];
                {
                    int ping_pong = ((compute_stage/STAGES)%2) * STAGES * (WN*BN);
                    int stage_in_window = compute_stage % STAGES;
                    int wg_start = (warp_id/4) * 4 * BN;
                    int warp_off = (warp_id%4) * 16 + (lane_id/4);
                    for (int tn = 0; tn < TN; tn++) {
                        int n_row_gate = wg_start + tn*64 + warp_off;
                        scale_w[tn][0] = s.scale_w_up[ping_pong + n_row_gate * STAGES + stage_in_window];
                        scale_w[tn][1] = s.scale_w_up[ping_pong + (n_row_gate + 8) * STAGES + stage_in_window];
                    }
                }

                float tile_acc[TN][TM][4];
                memset(tile_acc, 0, sizeof(tile_acc));
                for (int tn = 0; tn < TN; tn++)
                    for (int tm = 0; tm < TM; tm++)
                        for (int t = 0; t < 4; t++)
                            asm volatile("" : "+f"(tile_acc[tn][tm][t]) :: "memory");

                cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
                warpgroup_arrive();
                for(int tn = 0; tn<TN; tn++)
                {
                    fp8* sw_ptr = s.w_fp8 + dq_slot*WS + (warp_id/4)*(BN*4)*BK + tn*64*BK;
                    fp8* sx_ptr = s.x + ss*XS;
                    uint64_t desc_w = make_smem_descriptor<64, 1, S_MODE_UP>(sw_ptr);
                    uint64_t desc_x = make_smem_descriptor<64, 1, S_MODE_UP>(sx_ptr);
                    wgmma<1,1,1, BM>(tile_acc[tn], desc_w, desc_x);
                    wgmma<1,1,1, BM>(tile_acc[tn], desc_w+1*(32>>4), desc_x+1*(32>>4));
                    wgmma<1,1,1, BM>(tile_acc[tn], desc_w+2*(32>>4), desc_x+2*(32>>4));
                    wgmma<1,1,1, BM>(tile_acc[tn], desc_w+3*(32>>4), desc_x+3*(32>>4));
                }
                warpgroup_commit_batch();
                warpgroup_wait();

                arrive(bar + S + ss);

                for(int tm = 0; tm<TM; tm++)
                {
                    for(int tn = 0; tn<TN; tn++)
                    {
                        f_acc[tn][tm][0] = __hadd2(f_acc[tn][tm][0],
                                __nv_bfloat162(
                                    scale_w[tn][0] * scale_x[tm*2] * tile_acc[tn][tm][0],
                                    scale_w[tn][0] * scale_x[tm*2 + 1] * tile_acc[tn][tm][1]
                                    )
                                );
                        f_acc[tn][tm][1] = __hadd2(f_acc[tn][tm][1],
                                __nv_bfloat162(
                                    scale_w[tn][1] * scale_x[tm*2] * tile_acc[tn][tm][2],
                                    scale_w[tn][1] * scale_x[tm*2 + 1] * tile_acc[tn][tm][3]
                                    )
                                );
                    }
                }

                if (ss == STAGES - 1)
                    p_load ^= 1;
            }
        }

        // Release DOWN bar_release BEFORE SwiGLU so producer can start DOWN TMA loads
        // immediately, overlapping with consumer's SwiGLU + FP8 quantization.
        // Safe because producer writes s_d.w_packed (low SMEM) while consumer writes
        // s_d.x / s_d.out / token_scale_bridge (high SMEM) — no overlap.
        for (int i = 0; i < S; i++)
            arrive(&bar_down[S + i]);

        // SwiGLU + FP8 quantization
        consumer_sync();
        smem_down_w4a8<STAGES, WN, BM, BK, BN>& s_d = *reinterpret_cast<smem_down_w4a8<STAGES, WN, BM, BK, BN>*>(sh);
        float4* block_max = reinterpret_cast<float4*>(s_d.out);
        constexpr float EPS = 1e-10;
        nv_bfloat162 token_max[TM] = { nv_bfloat162(EPS, EPS) };
        for(int tn = 0; tn<TN; tn++)
        {
            for(int tm = 0; tm<TM; tm++)
            {
                f_acc[tn][tm][0].x = swiglu_mul(f_acc[tn][tm][0].x, f_acc[tn][tm][1].x);
                f_acc[tn][tm][0].y = swiglu_mul(f_acc[tn][tm][0].y, f_acc[tn][tm][1].y);
                nv_bfloat162 abs = __habs2(f_acc[tn][tm][0]);
                token_max[tm] = __hmax2(abs, token_max[tm]);
            }
        }
        for(int tm = 0; tm<TM; tm++)
        {
            token_max[tm] = __hmax2(__shfl_xor_sync(0xFFFFFFFF, token_max[tm], 16), token_max[tm]);
            token_max[tm] = __hmax2(__shfl_xor_sync(0xFFFFFFFF, token_max[tm], 8), token_max[tm]);
            token_max[tm] = __hmax2(__shfl_xor_sync(0xFFFFFFFF, token_max[tm], 4), token_max[tm]);
            if (lane_id < 4)
            {
                int off = tm*8 + (lane_id)*2;
                reinterpret_cast<nv_bfloat162*>(block_max + off * WARPGROUPS)[warp_id] = token_max[tm];
            }
        }
        consumer_sync();
        constexpr float fp8_max = 448.0;
        constexpr float fp8_min = -448.0;
        float token_scale[TM][2];

        for(int tm = 0; tm<TM; tm++)
        {
            int off = tm*8 + (lane_id%4)*2;
            for(int wg = 0; wg < WARPGROUPS; wg++)
            {
                float4 bmax = block_max[off * WARPGROUPS + wg];
                token_max[tm] = __hmax2(*reinterpret_cast<nv_bfloat162*>(&bmax.x), token_max[tm]);
                token_max[tm] = __hmax2(*reinterpret_cast<nv_bfloat162*>(&bmax.y), token_max[tm]);
                token_max[tm] = __hmax2(*reinterpret_cast<nv_bfloat162*>(&bmax.z), token_max[tm]);
                token_max[tm] = __hmax2(*reinterpret_cast<nv_bfloat162*>(&bmax.w), token_max[tm]);
            }
            token_scale[tm][0] = float(token_max[tm].x) / fp8_max;
            token_scale[tm][1] = float(token_max[tm].y) / fp8_max;
            for (int t = 0; t < 2; t++)
            {
                for(int tn = 0; tn<TN; tn++)
                {
                    float val = t == 0 ? f_acc[tn][tm][0].x : f_acc[tn][tm][0].y;
                    float q = val / token_scale[tm][t];
                    val = fminf(fmaxf(q, fp8_min), fp8_max);
                    int x_row = tm*8 + (lane_id%4)*2 + t;
                    int x_col = (warp_id/4)*(TN*32) + tn*32 + (warp_id%4)*8 + lane_id/4;
                    int idx = x_row*BK2 + x_col;
                    int swizzled = swizzle<S_BITS_DOWN>(idx);
                    s_d.x[swizzled] = fp8(val);
                }
            }
        }

        // Store token_scale to bridge for down projection consumer
        // Write to smem_down_w4a8's tail to ensure DOWN producer TMA won't overwrite it
        for(int tm = 0; tm<TM; tm++)
        {
            for (int t = 0; t < 2; t++)
            {
                int row = tm*8 + (lane_id%4)*2 + t;
                smem_down_w4a8<STAGES, WN, BM, BK, BN>& s_bridge = *reinterpret_cast<smem_down_w4a8<STAGES, WN, BM, BK, BN>*>(sh);
                s_bridge.token_scale_bridge[row] = token_scale[tm][t];
            }
        }

        for(int t = 0; t < TPT; t+=2)
        {
            int token_idx = (t/2)*8 + (lane_id%4)*2;
            const float2 topk_w = *reinterpret_cast<const float2*>(&topk_scales[token_idx]);
            token_scale[t/2][0] *= topk_w.x * scaling_factor;
            token_scale[t/2][1] *= topk_w.y * scaling_factor;
        }

        consumer_sync();

        // ========== DOWN PROJECTION (w4a8: packed INT4 + dequant) ==========
        // No transition barriers needed! Producer already started DOWN TMA loads
        // using separate bar_down[] barriers, overlapping with consumer's SwiGLU.
        // DOWN CONSUMER: pipelined dequant/WGMMA overlap (STAGES>=2) or sequential (STAGES==1)
        // For DOWN, WGMMA reads w_fp8[dq_slot] + s_d.x (fixed, not reused by producer).
        // Producer only writes w_packed[ss] + scale_w_down, so overlap is safe for STAGES>=2.
        {
        // Read token_scale from bridge (stored in smem_down_w4a8 tail)
        smem_down_w4a8<STAGES, WN, BM, BK, BN>& s_bridge = *reinterpret_cast<smem_down_w4a8<STAGES, WN, BM, BK, BN>*>(sh);
        float token_scale[TM][2];
        for(int tm = 0; tm<TM; tm++)
        {
            for (int t = 0; t < 2; t++)
            {
                int row = tm*8 + (lane_id%4)*2 + t;
                token_scale[tm][t] = s_bridge.token_scale_bridge[row];
            }
        }

        // Apply topk_weights and scaling_factor
        for(int t = 0; t < TPT; t+=2)
        {
            int token_idx = (t/2)*8 + (lane_id%4)*2;
            const float2 topk_w = *reinterpret_cast<const float2*>(&topk_scales[token_idx]);
            token_scale[t/2][0] *= topk_w.x * scaling_factor;
            token_scale[t/2][1] *= topk_w.y * scaling_factor;
        }

        int token_src = M;
        if(threadIdx.x < PRODUCER_THREADS + BM)
            token_src = sorted_token_ids[warpM*BM + threadIdx.x-PRODUCER_THREADS] / top_k;

        // Consumer-side dequant constants for DOWN
        constexpr int CONSUMER_DQ_PER_THREAD_DOWN = TOTAL_UINT32_DOWN / CONSUMER_THREADS;
        const int cons_tid = threadIdx.x - PRODUCER_THREADS;

        // Precompute FP8 LUT
        uint32_t neg_lo, neg_hi, pos_lo, pos_hi;
        make_fp8_lut(1.0f, neg_lo, neg_hi, pos_lo, pos_hi);

        // DOWN bar_release initial arrive already done before SwiGLU (overlapped with producer).
        // No need to arrive again here.

        smem_down_w4a8<STAGES, WN, BM, BK, BN>& s_d = *reinterpret_cast<smem_down_w4a8<STAGES, WN, BM, BK, BN>*>(sh);

        // Lambda: dequant a DOWN stage into w_fp8[dq_slot]
        // uint4 (16B) merged writes: same pair-based strategy as UP dequant.
        auto do_dequant_down = [&](int smem_slot, int dq_slot) __attribute__((always_inline)) {
            uint32_t* src = reinterpret_cast<uint32_t*>(s_d.w_packed + smem_slot * PACKED_WS_DOWN);
            constexpr int WARP_SIZE = 32;
            constexpr int PAIRS_PER_THREAD_DOWN = CONSUMER_DQ_PER_THREAD_DOWN / 2;
            const int warp_in_cons = cons_tid / WARP_SIZE;
            const int lane_in_warp = cons_tid % WARP_SIZE;
            const int pair_base = warp_in_cons * (WARP_SIZE * PAIRS_PER_THREAD_DOWN) + lane_in_warp;
            // Process 4 pairs per iteration for maximum ILP
            #pragma unroll
            for (int r = 0; r < PAIRS_PER_THREAD_DOWN; r += 4)
            {
                int pidx0 = pair_base + r * WARP_SIZE;
                int pidx1 = pidx0 + WARP_SIZE;
                int pidx2 = pidx0 + 2 * WARP_SIZE;
                int pidx3 = pidx0 + 3 * WARP_SIZE;
                uint32_t packed0e = src[pidx0 * 2];
                uint32_t packed0o = src[pidx0 * 2 + 1];
                uint32_t packed1e = src[pidx1 * 2];
                uint32_t packed1o = src[pidx1 * 2 + 1];
                uint32_t packed2e = src[pidx2 * 2];
                uint32_t packed2o = src[pidx2 * 2 + 1];
                uint32_t packed3e = src[pidx3 * 2];
                uint32_t packed3o = src[pidx3 * 2 + 1];
                uint32_t lo0e, hi0e, lo0o, hi0o;
                uint32_t lo1e, hi1e, lo1o, hi1o;
                uint32_t lo2e, hi2e, lo2o, hi2o;
                uint32_t lo3e, hi3e, lo3o, hi3o;
                dequant_8xint4_to_8xfp8(packed0e, neg_lo, neg_hi, pos_lo, pos_hi, lo0e, hi0e);
                dequant_8xint4_to_8xfp8(packed0o, neg_lo, neg_hi, pos_lo, pos_hi, lo0o, hi0o);
                dequant_8xint4_to_8xfp8(packed1e, neg_lo, neg_hi, pos_lo, pos_hi, lo1e, hi1e);
                dequant_8xint4_to_8xfp8(packed1o, neg_lo, neg_hi, pos_lo, pos_hi, lo1o, hi1o);
                dequant_8xint4_to_8xfp8(packed2e, neg_lo, neg_hi, pos_lo, pos_hi, lo2e, hi2e);
                dequant_8xint4_to_8xfp8(packed2o, neg_lo, neg_hi, pos_lo, pos_hi, lo2o, hi2o);
                dequant_8xint4_to_8xfp8(packed3e, neg_lo, neg_hi, pos_lo, pos_hi, lo3e, hi3e);
                dequant_8xint4_to_8xfp8(packed3o, neg_lo, neg_hi, pos_lo, pos_hi, lo3o, hi3o);
                int fp8_off = dq_slot * WS_DOWN;
                int sw0 = swizzle<S_BITS_DOWN>(fp8_off + (pidx0 * 2) * 8);
                int sw1 = swizzle<S_BITS_DOWN>(fp8_off + (pidx1 * 2) * 8);
                int sw2 = swizzle<S_BITS_DOWN>(fp8_off + (pidx2 * 2) * 8);
                int sw3 = swizzle<S_BITS_DOWN>(fp8_off + (pidx3 * 2) * 8);
                *reinterpret_cast<uint4*>(s_d.w_fp8 + sw0) = make_uint4(lo0e, hi0e, lo0o, hi0o);
                *reinterpret_cast<uint4*>(s_d.w_fp8 + sw1) = make_uint4(lo1e, hi1e, lo1o, hi1o);
                *reinterpret_cast<uint4*>(s_d.w_fp8 + sw2) = make_uint4(lo2e, hi2e, lo2o, hi2o);
                *reinterpret_cast<uint4*>(s_d.w_fp8 + sw3) = make_uint4(lo3e, hi3e, lo3o, hi3o);
            }
        };

        // Macro for DOWN post-WGMMA: scale, st_matrix, cp_reduce_async
        // (shared between pipelined and sequential paths)
        #define DOWN_POST_WGMMA(compute_stage, ss, dq_slot) \
        { \
            arrive(bar_down + S + ss); \
            constexpr int PAD = BN2+8; \
            __nv_bfloat16 out_tile[TN2/2][TM][8]; \
            for(int tn2 = 0; tn2<TN2; tn2+=2) \
            { \
                for(int tm = 0; tm<TM; tm++) \
                { \
                    for (int t = 0; t<8; t++) \
                    { \
                        out_tile[tn2/2][tm][t] = token_scale[tm][t%2] * tile_acc[tn2 + t/4][tm][t%4] * s_w[tn2 + t/4][(t%4)/2]; \
                    } \
                } \
            } \
            asm volatile("cp.async.bulk.wait_group 0;"); \
            cuda::ptx::fence_proxy_async(cuda::ptx::space_shared); \
            consumer_sync(); \
            for(int tn2 = 0; tn2<TN2; tn2+=2) \
            { \
                for(int tm = 0; tm<TM; tm++) \
                { \
                    int out_row = tm * 8 + lane_id%8; \
                    int out_col = (warp_id/4)*TN2*64 + (warp_id%4)*16 + (lane_id&8) + tn2*64 + (lane_id/16)*64; \
                    st_matrix_x4_trans(reinterpret_cast<uint32_t*>(out_tile[tn2/2][tm]), \
                            __cvta_generic_to_shared(s_d.out + out_row*PAD + out_col)); \
                } \
            } \
            cuda::ptx::fence_proxy_async(cuda::ptx::space_shared); \
            consumer_sync(); \
            if(threadIdx.x < PRODUCER_THREADS + BM) \
            { \
                if (token_src < M) \
                { \
                    int row = threadIdx.x - PRODUCER_THREADS; \
                    cuda::ptx::cp_reduce_async_bulk( \
                            cuda::ptx::space_global, \
                            cuda::ptx::space_shared, \
                            cuda::ptx::op_add, \
                            out + token_src*N2 + compute_stage*BN2, \
                            s_d.out + row*PAD, \
                            BN2*sizeof(__nv_bfloat16)); \
                } \
                cuda::ptx::cp_async_bulk_commit_group(); \
            } \
        }

        int p_load = 0;

        if constexpr (STAGES >= 2)
        {
            // ---- Pipelined mode: prologue dequant stage 0, then overlap ----
            {
                wait(bar_down + 0, p_load);
                do_dequant_down(0, 0);
                // No dequant_sync: wgmma.fence.sync.aligned handles warpgroup synchronization
            }

            for (int compute_stage = 0; compute_stage < n_stages_down; compute_stage += 1)
            {
                int ss = compute_stage % STAGES;
                int dq_slot = compute_stage % 2;

                // per-row: each tn2 takes two scales (i=0,1 row and i=2,3 row=+8)
                float s_w[TN2][2];
                float tile_acc[TN2][TM][4];
                memset(tile_acc, 0, sizeof(tile_acc));
                fp8* sx = s_d.x;

                for (int tn = 0; tn < TN2; tn++)
                    for (int tm = 0; tm < TM; tm++)
                        for (int t = 0; t < 4; t++)
                            asm volatile("" : "+f"(tile_acc[tn][tm][t]) :: "memory");

                cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
                warpgroup_arrive();
                {
                    // SMEM scale base for this stage:
                    //   ping/pong:           ((compute_stage/STAGES)%2) * (STAGES*BN2)
                    //   stage within window: (compute_stage%STAGES) * BN2
                    //   per-lane wgmma N2-row mapping (mirrors UP, see ptx 9.7.15.5.8):
                    //     wg_start  = (warp_id/4) * TN2 * 64
                    //     warp_off  = (warp_id%4) * 16 + (lane_id/4)
                    //     row_i01   = wg_start + tn2*64 + warp_off
                    //     row_i23   = row_i01 + 8
                    // Iteration 2: SMEM layout is [ping_pong][row][stage_in_window], stride STAGES
                    const int ping_pong       = ((compute_stage/STAGES)%2) * (STAGES*BN2);
                    const int stage_in_window = compute_stage % STAGES;
                    const int row_base        = (warp_id/4) * TN2 * 64
                                              + (warp_id%4) * 16
                                              + (lane_id/4);
                    for (int tn2_i = 0; tn2_i < TN2; tn2_i++)
                    {
                        int row01 = row_base + tn2_i*64;
                        int row23 = row01 + 8;
                        s_w[tn2_i][0] = scale_w_down[ping_pong + row01 * STAGES + stage_in_window];
                        s_w[tn2_i][1] = scale_w_down[ping_pong + row23 * STAGES + stage_in_window];
                    }
                }

                for(int tn2 = 0; tn2 < TN2; tn2++)
                {
                    fp8* sw = s_d.w_fp8 + dq_slot*WS_DOWN + ((warp_id/4)*TN2 + tn2)*64*BK2;
                    uint64_t desc_w = make_smem_descriptor<BK2/2, 1, S_MODE_DOWN>(sw);
                    uint64_t desc_x = make_smem_descriptor<BK2/2, 1, S_MODE_DOWN>(sx);
                    for(int tk = 0; tk < BK2/32; tk++)
                    {
                        wgmma<1,1,1, BM>(tile_acc[tn2], desc_w, desc_x);
                        desc_w += (32>>4);
                        desc_x += (32>>4);
                    }
                }
                warpgroup_commit_batch();

                // Overlap: dequant next stage while WGMMA is in-flight
                if (compute_stage + 1 < n_stages_down)
                {
                    int next_ss = (compute_stage + 1) % STAGES;
                    int next_dq_slot = (compute_stage + 1) % 2;
                    int next_p_load = p_load;
                    if (next_ss == 0)
                        next_p_load = p_load ^ 1;

                    wait(bar_down + next_ss, next_p_load);
                    do_dequant_down(next_ss, next_dq_slot);
                    // No dequant_sync: wgmma.fence.sync.aligned in next iteration handles it
                }

                warpgroup_wait();
                DOWN_POST_WGMMA(compute_stage, ss, dq_slot);

                if (ss == STAGES - 1)
                    p_load ^= 1;
            }
        }
        else  // STAGES == 1: sequential dequant→WGMMA
        {
            for (int compute_stage = 0; compute_stage < n_stages_down; compute_stage += 1)
            {
                int ss = compute_stage % STAGES;
                int dq_slot = compute_stage % 2;

                wait(bar_down + ss, p_load);
                do_dequant_down(ss, dq_slot);
                // No dequant_sync: wgmma.fence.sync.aligned below handles it

                // per-row: each tn2 takes two scales (i=0,1 row and i=2,3 row=+8)
                float s_w[TN2][2];
                float tile_acc[TN2][TM][4];
                memset(tile_acc, 0, sizeof(tile_acc));
                fp8* sx = s_d.x;

                for (int tn = 0; tn < TN2; tn++)
                    for (int tm = 0; tm < TM; tm++)
                        for (int t = 0; t < 4; t++)
                            asm volatile("" : "+f"(tile_acc[tn][tm][t]) :: "memory");

                cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
                warpgroup_arrive();
                {
                    // Iteration 2: SMEM layout is [ping_pong][row][stage_in_window], stride STAGES
                    const int ping_pong       = ((compute_stage/STAGES)%2) * (STAGES*BN2);
                    const int stage_in_window = compute_stage % STAGES;
                    const int row_base        = (warp_id/4) * TN2 * 64
                                              + (warp_id%4) * 16
                                              + (lane_id/4);
                    for (int tn2_i = 0; tn2_i < TN2; tn2_i++)
                    {
                        int row01 = row_base + tn2_i*64;
                        int row23 = row01 + 8;
                        s_w[tn2_i][0] = scale_w_down[ping_pong + row01 * STAGES + stage_in_window];
                        s_w[tn2_i][1] = scale_w_down[ping_pong + row23 * STAGES + stage_in_window];
                    }
                }

                for(int tn2 = 0; tn2 < TN2; tn2++)
                {
                    fp8* sw = s_d.w_fp8 + dq_slot*WS_DOWN + ((warp_id/4)*TN2 + tn2)*64*BK2;
                    uint64_t desc_w = make_smem_descriptor<BK2/2, 1, S_MODE_DOWN>(sw);
                    uint64_t desc_x = make_smem_descriptor<BK2/2, 1, S_MODE_DOWN>(sx);
                    for(int tk = 0; tk < BK2/32; tk++)
                    {
                        wgmma<1,1,1, BM>(tile_acc[tn2], desc_w, desc_x);
                        desc_w += (32>>4);
                        desc_x += (32>>4);
                    }
                }
                warpgroup_commit_batch();
                warpgroup_wait();
                DOWN_POST_WGMMA(compute_stage, ss, dq_slot);

                if (ss == STAGES - 1)
                    p_load ^= 1;
            }
        }
        #undef DOWN_POST_WGMMA
        }
    }
}

// ============================================================================
// w4a8 launch / dispatch functions
// ============================================================================

// Packed forward-call args shared by the 4 dispatch layers below.
struct FusedMoeW4A8Args {
    const fp8* x;
    const float* x_scale;
    uint8_t* w;                 const float* w_scale;
    uint8_t* w2;                const float* w2_scale;
    __nv_bfloat16* out;
    const int* sorted_token_ids;
    const int* expert_ids;
    const int* num_tokens_post_padded;
    const float* topk_weights;
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

template<int BM, int BN, int WN, int STAGES>
void launch_fused_moe_w4a8_kernel_up_down_acc(const FusedMoeW4A8Args& a)
{
    constexpr int BK = 128;
    constexpr int PRODUCER_THREADS = 128;
    dim3 dimBlock(32*WN + PRODUCER_THREADS, 1, 1);
    dim3 dimGrid(std::ceil((float)a.N/(BN*WN)),
                 std::ceil((float)a.sorted_num/(a.block_m)), 1);

    // smem = max(smem_up_w4a8, smem_down_w4a8)
    size_t sMemSize = std::max(sizeof(smem_up_w4a8<STAGES, WN, BM, BK, BN>),
                               sizeof(smem_down_w4a8<STAGES, WN, BM, BK, BN>));
    // H100 max dynamic SMEM per block ≈ 227 KB. Configs exceeding this
    // make cudaFuncSetAttribute return cudaErrorInvalidValue. Turn that
    // into a std::runtime_error (caught by the Python op layer) so the
    // JIT tuner can sweep the full bm range without subprocess deaths.
    cudaError_t smem_err = cudaFuncSetAttribute(
        fused_moe_w4a8_wgmma_up_down_acc_kernel<BM, BK, BN, WN, STAGES, PRODUCER_THREADS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize);
    if (smem_err != cudaSuccess) {
        cudaGetLastError();  // clear sticky error
        char buf[256];
        snprintf(buf, sizeof(buf),
            "fused_moe_w4a8 cudaFuncSetAttribute failed (SMEM=%zu B) "
            "(BM=%d BN=%d WN=%d STAGES=%d): %s",
            sMemSize, BM, BN, WN, STAGES, cudaGetErrorString(smem_err));
        throw std::runtime_error(buf);
    }

    // TMA descriptors for the INT4-packed weights (UP + DOWN). Both are
    // 2D UINT8 tiles with no swizzle; only shape/box differ.
    auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();
    auto make_int4_packed_tma = [&](CUtensorMap* map, uint8_t* ptr,
                                    uint64_t dim0, uint64_t dim1,
                                    uint32_t box0, uint32_t box1) {
        uint64_t size[2]        = {dim0, dim1};
        uint64_t stride[1]      = {dim0 * sizeof(uint8_t)};
        uint32_t box_size[2]    = {box0, box1};
        uint32_t elem_stride[2] = {1, 1};
        cuTensorMapEncodeTiled(
                map,
                CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
                /*rank=*/2, ptr, size, stride, box_size, elem_stride,
                CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
                CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
                CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    };

    // UP: [K/2, N*num_experts], box = [BK/2, BN*WN]
    CUtensorMap tensor_map_w_packed{};
    make_int4_packed_tma(&tensor_map_w_packed, a.w,
                         static_cast<uint64_t>(a.K/2),
                         static_cast<uint64_t>(a.N*a.num_experts),
                         static_cast<uint32_t>(BK/2),
                         static_cast<uint32_t>(BN*WN));

    // DOWN: [K2/2, N2*num_experts], box = [BK2/2, BN2]
    constexpr int BK2 = WN*BN/2;
    constexpr int BN2 = BK*2;
    const int K2 = a.N/2;
    const int N2 = a.K;
    CUtensorMap tensor_map_w2_packed{};
    make_int4_packed_tma(&tensor_map_w2_packed, a.w2,
                         static_cast<uint64_t>(K2/2),
                         static_cast<uint64_t>(N2*a.num_experts),
                         static_cast<uint32_t>(BK2/2),
                         static_cast<uint32_t>(BN2));

    fused_moe_w4a8_wgmma_up_down_acc_kernel<BM, BK, BN, WN, STAGES, PRODUCER_THREADS>
        <<<dimGrid, dimBlock, sMemSize, a.stream>>>(
            a.x, a.x_scale,
            tensor_map_w_packed, a.w_scale,
            tensor_map_w2_packed, a.w2_scale,
            a.out,
            a.sorted_token_ids, a.expert_ids,
            a.num_tokens_post_padded, a.topk_weights,
            a.top_k, a.M, a.K, a.N,
            a.scaling_factor);
}

template<int BM, int BN, int WN>
void dispatch_stages_w4a8(const FusedMoeW4A8Args& a)
{
    switch (a.stages) {
        case 1: launch_fused_moe_w4a8_kernel_up_down_acc<BM, BN, WN, 1>(a); break;
        case 2: launch_fused_moe_w4a8_kernel_up_down_acc<BM, BN, WN, 2>(a); break;
        case 3: launch_fused_moe_w4a8_kernel_up_down_acc<BM, BN, WN, 3>(a); break;
        case 4: launch_fused_moe_w4a8_kernel_up_down_acc<BM, BN, WN, 4>(a); break;
        case 5: launch_fused_moe_w4a8_kernel_up_down_acc<BM, BN, WN, 5>(a); break;
        default: fprintf(stderr, "Unsupported stages value: %d for w4a8\n", a.stages);
    }
}

template<int BM>
void dispatch_bn_wn_w4a8(const FusedMoeW4A8Args& a)
{
    if      (a.block_n == 32 && a.warp_n == 8) dispatch_stages_w4a8<BM, 32, 8>(a);
    else if (a.block_n == 64 && a.warp_n == 4) dispatch_stages_w4a8<BM, 64, 4>(a);
    else fprintf(stderr, "Unsupported BN/WN pair: (%d, %d) for w4a8\n",
                 a.block_n, a.warp_n);
}

void fused_moe_w4a8_wgmma_up_down_acc(const FusedMoeW4A8Args& a)
{
    switch (a.block_m) {
        case   8: dispatch_bn_wn_w4a8<  8>(a); break;
        case  16: dispatch_bn_wn_w4a8< 16>(a); break;
        case  24: dispatch_bn_wn_w4a8< 24>(a); break;
        case  32: dispatch_bn_wn_w4a8< 32>(a); break;
        case  40: dispatch_bn_wn_w4a8< 40>(a); break;
        case  48: dispatch_bn_wn_w4a8< 48>(a); break;
        case  56: dispatch_bn_wn_w4a8< 56>(a); break;
        case  64: dispatch_bn_wn_w4a8< 64>(a); break;
        case  72: dispatch_bn_wn_w4a8< 72>(a); break;
        case  80: dispatch_bn_wn_w4a8< 80>(a); break;
        case  88: dispatch_bn_wn_w4a8< 88>(a); break;
        case  96: dispatch_bn_wn_w4a8< 96>(a); break;
        case 104: dispatch_bn_wn_w4a8<104>(a); break;
        case 112: dispatch_bn_wn_w4a8<112>(a); break;
        case 120: dispatch_bn_wn_w4a8<120>(a); break;
        case 128: dispatch_bn_wn_w4a8<128>(a); break;
        default:  fprintf(stderr, "Unsupported block_m value: %d for w4a8\n",
                          a.block_m);
    }
}
