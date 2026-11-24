#pragma once

#include <algorithm>

#include <deep_gemm/layout/mega_moe_fused.cuh>

#include "../../utils/exception.hpp"
#include "sm90.hpp"

namespace deep_gemm {

static constexpr int kSM90NVFP4BStoragePerKBlock = 80;

struct SM90FP4H20FusedConfig {
    static constexpr int kBlockK = 128;
    static constexpr int kSwizzleActsMode = 128;
    static constexpr int kNumDispatchThreads = 64;
    static constexpr int kNumNonEpilogueThreads = 64;
    static constexpr int kNumEpilogueThreads = 256;
    static constexpr int kNumThreads =
        kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads;

    int block_m, block_n;
    int num_max_pool_tokens;
    int num_padded_sf_pool_tokens;
    int num_experts_per_wave;
    int num_stages, smem_size;
};

struct SM90FP4H20FusedShape {
    static constexpr int kH200NumSMs = 78;
    static constexpr int kNumRanks = 8;
    static constexpr int kExpertsPerRank = 48;
    // One node of a two-node EP16: 192 experts over the 8 NVLink ranks.
    static constexpr int kEP16ExpertsPerRank = 24;
    static constexpr int kTopk = 8;
    static constexpr int kHidden = 3072;
    static constexpr int kIntermediateHidden = 1280;

    int num_sms;
    int num_ranks;
    int num_experts;
    int num_topk;
    int hidden;
    int intermediate_hidden;

    static constexpr bool is_supported_batch(const int num_tokens) noexcept {
        return num_tokens > 0;
    }

    constexpr bool is_supported_h200_shape() const noexcept {
        // Fewer SMs than the device has: EP16 overlap keeps SMs free for the
        // concurrent NCCL kernel (see deep_gemm/mega/ep16.py).
        return num_sms <= kH200NumSMs && num_sms >= kH200NumSMs - 8 &&
            num_ranks == kNumRanks &&
            (num_experts == kExpertsPerRank * kNumRanks ||
             num_experts == kEP16ExpertsPerRank * kNumRanks) &&
            num_topk == kTopk &&
            hidden == kHidden &&
            intermediate_hidden == kIntermediateHidden;
    }
};

struct SM90FP4H20FusedInput {
    int num_sms;
    int num_ranks, num_experts, num_experts_per_rank;
    int num_max_tokens_per_rank, num_tokens, num_topk;
    int hidden, intermediate_hidden;
    int num_padded_sf_pool_tokens;
    // RF-decode swapAB hosts (MXFP4, QoQ; kernel `kRFDecode`) run multi-K-block
    // BM8 stages; NVFP4 keeps one K-block per stage (>= 4 stages).
    bool rf_decode = false;
    // Wide (2 packed tiles == 512-row) L1 and/or L2 tasks requested for this launch
    // (host env DG_FP4_L1_BN / DG_FP4_L2_BN, see the host): the BM8 RF tier then
    // carries ONE K128 block per stage (1 KB A + 40 KB B) in >= 4 stages.
    bool wide_tiles = false;

    SM90FP4H20FusedShape shape() const noexcept {
        return {
            num_sms, num_ranks, num_experts, num_topk,
            hidden, intermediate_hidden};
    }
};

struct SM90FP4H20FusedPlan {
    SM90FP4H20FusedConfig config;
    bool swap_ab;
    bool use_mode2_row_decoder;
    bool single_active_dispatch_warp;
    bool use_interleaved_scheduler;
};

static SM90FP4H20FusedPlan
select_sm90_nvfp4_h200_fused(
        const SM90FP4H20FusedInput& input) {
    DG_HOST_ASSERT(input.shape().is_supported_h200_shape());
    DG_HOST_ASSERT(
        input.num_experts_per_rank == SM90FP4H20FusedShape::kExpertsPerRank ||
        input.num_experts_per_rank == SM90FP4H20FusedShape::kEP16ExpertsPerRank);
    DG_HOST_ASSERT(input.num_experts ==
                   input.num_experts_per_rank * input.num_ranks);
    DG_HOST_ASSERT(input.num_max_tokens_per_rank > 0);
    DG_HOST_ASSERT(input.num_tokens <= input.num_max_tokens_per_rank);
    DG_HOST_ASSERT(
        SM90FP4H20FusedShape::is_supported_batch(input.num_tokens));
    DG_HOST_ASSERT(input.num_padded_sf_pool_tokens > 0);

    struct Tuning {
        int block_m, block_n;
        int num_experts_per_wave;
        int num_stages;
        int smem_size;
        bool swap_ab;
        bool use_mode2_row_decoder;
        bool single_active_dispatch_warp;
    } tuning {};

    // BM8 (MXFP4 / QoQ RF swapAB) runs kKBlocksPerStage == 2: a stage carries 2 K128
    // blocks (2 KB A + 40 KB packed B + 2 SFA slots, 4 stages = 173056 B); wide tasks
    // carry one K128 block (1 KB A + 2 x 20 KB B) per stage. DG_FP4_BM8_STAGES overrides
    // the depth (kernel static-asserts 2..4 for 2 blocks, 4..7 for 1 block).
    const int bm8_k_blocks = (input.rf_decode && !input.wide_tiles) ? 2 : 1;
    const int bm8_stages =
        std::clamp(get_env<int>("DG_FP4_BM8_STAGES", 4), bm8_k_blocks == 2 ? 2 : 4, bm8_k_blocks == 2 ? 4 : 7);
    if (input.num_tokens <= 1)
        tuning = {8, 256, 24, bm8_stages, SM90ArchSpec::smem_capacity,
                  true, true, true};
    else if (input.num_tokens <= 8)
        // EP16 (24 experts per rank): one wave of all 24
        tuning = {8, 256, input.num_experts_per_rank % 16 == 0 ? 16 : 24, bm8_stages,
                  SM90ArchSpec::smem_capacity, true, true, true};
    else if (input.num_tokens <= 16)
        tuning = {8, 256, 24, bm8_stages, SM90ArchSpec::smem_capacity,
                  true, true, true};
    else if (input.num_tokens <= 32)
        tuning = {16, 256, 48, 3, SM90ArchSpec::smem_capacity,
                  true, true, false};
    else if (input.num_tokens <= 64)
        tuning = {24, 256, 48, 3, 229312,
                  true, false, true};
    else if (input.num_tokens <= 256)
        tuning = {64, 256, 48, 3, 209856,
                  false, true, false};
    else
        tuning = {128, 128, 48, 6, SM90ArchSpec::smem_capacity,
                  false, true, false};

    DG_HOST_ASSERT(
        input.num_experts_per_rank % tuning.num_experts_per_wave == 0);
    DG_HOST_ASSERT(tuning.smem_size <= SM90ArchSpec::smem_capacity);
    return {
        {
            tuning.block_m,
            tuning.block_n,
            fused_layout::get_num_max_pool_tokens(
                input.num_ranks, input.num_max_tokens_per_rank,
                input.num_topk, input.num_experts_per_rank),
            input.num_padded_sf_pool_tokens,
            tuning.num_experts_per_wave,
            tuning.num_stages,
            cute::min(tuning.smem_size +
                          fused_layout::kSM90InterleavedSchedulerSMEMBytes,
                      SM90ArchSpec::smem_capacity),
        },
        tuning.swap_ab,
        tuning.use_mode2_row_decoder,
        tuning.single_active_dispatch_warp,
        true,
    };
}

}  // namespace deep_gemm
